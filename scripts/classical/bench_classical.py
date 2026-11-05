# /// script
# requires-python = ">=3.11"
# ///
"""Kev.benchmark --data with a classical supervised baseline as the predictor: TF-IDF + logistic regression on the text tasks, a gradient-boosted tree ensemble on the structured tasks,
each trained on the same 3k subset as the fine-tuned rows and scored through kev.benchmark unchanged, so rows.json,
report.json and kev.calibrate work exactly as for Kev-4B and LoRA-SFT.

Features from the record state: numbers become numeric columns, short strings categorical columns, and long strings,
lists and free text one text field (word 1-2 gram TF-IDF). Logistic regression (multinomial, L2) sees the sparse
one-hot + TF-IDF matrix; the tree model (HistGradientBoosting) sees numeric and ordinal columns plus a 64-dimensional
SVD of the TF-IDF. One classifier per question id. The regularisation strength and, for the structured tasks, the model
family are chosen on the development split by macro-F1 (the same split on which every row's temperature is fitted).
Probabilities are the classifier's own; a label never seen in training gets 1e-6 before renormalisation. Rows carry
log-probabilities as logits and inference_temperature 1.0, so kev.calibrate fits a temperature as for the other rows.
The training seed changes only the tree ensemble; logistic regression is deterministic. Reads no row content aloud.

Usage: uv run --no-sync python scripts/classical/bench_classical.py --train data/derived/p2-3k/train.jsonl --task p2
           --seed 0 [--select runs/<dev out>] <kev.benchmark arguments: --data ... --out ...>
--select <dev out dir of the same seed>: reuse the family/C chosen on dev (written to <dev out>/classical_choice.json) for the test pass."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontier"))
from serve import options  # noqa: E402  (option keys exactly as every other generative/decision row uses them)

C_GRID = (0.3, 1.0, 3.0)
STRUCTURED = {"p3", "i1"}


def flatten(state, prefix="", num=None, cat=None, text=None):
    num = {} if num is None else num; cat = {} if cat is None else cat; text = [] if text is None else text
    if isinstance(state, dict):
        for k, v in state.items():
            flatten(v, f"{prefix}{k}.", num, cat, text)
    elif isinstance(state, list):
        text.append(" ".join(str(x) for x in state))
    elif isinstance(state, bool) or state is None:
        cat[prefix.rstrip(".")] = str(state)
    elif isinstance(state, (int, float)):
        num[prefix.rstrip(".")] = float(state)
    else:
        s = str(state)
        try:
            num[prefix.rstrip(".")] = float(s.replace(",", "."))
        except ValueError:
            if len(s) <= 40 and " " not in s.strip():
                cat[prefix.rstrip(".")] = s
            else:
                text.append(f"{prefix.rstrip('.')}: {s}" if prefix else s)
    return num, cat, " ".join(text)


def label_key(q):
    t, y = q["type"], q["label"]
    return ("true" if y else "false") if t == "noul" else str(y)


class ClassicalPredictor:
    task = None
    train = None
    seed = 0
    select = None
    data = None          # the file being scored: predictions are computed for all its records at once (batched), then looked up

    def __init__(self, run, device, opts=None, context=None, model=None, tok=None):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import OrdinalEncoder
        from scipy import sparse
        t0 = time.time()
        records = [json.loads(l) for l in open(self.train) if l.strip()]
        feats = [flatten(r["state"]) for r in records]
        self.dv_num = DictVectorizer(sparse=True); self.dv_cat = DictVectorizer(sparse=True)
        Xn = self.dv_num.fit_transform([f[0] for f in feats]); Xc = self.dv_cat.fit_transform([{k: v for k, v in f[1].items()} for f in feats])
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        Xt = self.tfidf.fit_transform([f[2] for f in feats])
        self.num_mean = np.asarray(Xn.mean(axis=0)).ravel(); self.num_std = np.sqrt(np.asarray(Xn.multiply(Xn).mean(axis=0)).ravel() - self.num_mean ** 2) + 1e-6
        Xn_s = sparse.csr_matrix((Xn.toarray() - self.num_mean) / self.num_std) if Xn.shape[1] else Xn
        self.X_sparse_train = sparse.hstack([Xn_s, Xc, Xt]).tocsr()
        self.qids = list(records[0]["questions"])
        self.qspec = {q: records[0]["questions"][q] for q in self.qids}
        self.keys = {q: options(self.qspec[q])[0] for q in self.qids}
        self.y = {q: np.array([self.keys[q].index(label_key(r["questions"][q])) if q in r["questions"] else -1 for r in records]) for q in self.qids}
        self.structured = self.task in STRUCTURED
        if self.structured:
            from sklearn.decomposition import TruncatedSVD
            self.svd = TruncatedSVD(n_components=min(64, Xt.shape[1] - 1), random_state=self.seed).fit(Xt)
            self.oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(Xc.toarray())
            self.X_dense_train = np.hstack([Xn.toarray(), self.oe.transform(Xc.toarray()), self.svd.transform(Xt)])
        self.models = {}
        choice = json.loads(Path(self.select, "classical_choice.json").read_text()) if self.select else None
        self.choice = choice or {}
        for q in self.qids:
            mask = self.y[q] >= 0
            cands = []
            if choice is None:
                for C in C_GRID:
                    cands.append(("lr", C))
                if self.structured:
                    for lr in (0.05, 0.1):
                        cands.append(("hgb", lr))
            else:
                cands = [tuple(choice[q])]
            self.models[q] = [(fam, hp, self._fit(fam, hp, q, mask)) for fam, hp in cands]
        self.temperature = 1.0
        self.fit_seconds = time.time() - t0
        self.cache = {}
        if self.data:
            self._precompute(self.data)

    @staticmethod
    def record_key(record):
        meta = record.get("_meta") or {}
        return str(meta.get("id") or hash(json.dumps(record["state"], sort_keys=True, ensure_ascii=False)))

    def _batch_features(self, states):
        from scipy import sparse
        fl = [flatten(s) for s in states]
        Xn = self.dv_num.transform([f[0] for f in fl]); Xc = self.dv_cat.transform([f[1] for f in fl]); Xt = self.tfidf.transform([f[2] for f in fl])
        Xn_s = sparse.csr_matrix((Xn.toarray() - self.num_mean) / self.num_std) if Xn.shape[1] else Xn
        xs = sparse.hstack([Xn_s, Xc, Xt]).tocsr()
        xd = np.hstack([Xn.toarray(), self.oe.transform(Xc.toarray()), self.svd.transform(Xt)]) if self.structured else None
        return xs, xd, np.asarray(Xt.sum(axis=1)).ravel()

    def _precompute(self, path):
        """All candidates' probabilities for every record of `path`, in one predict_proba call per candidate."""
        t0 = time.time()
        records = [json.loads(l) for l in open(path) if l.strip()]
        keys = [self.record_key(r) for r in records]
        xs, xd, ntok = self._batch_features([r["state"] for r in records])
        for q in self.qids:
            per_cand = []
            for fam, hp, model in self.models[q]:
                pr = model.predict_proba(xs if fam == "lr" else xd)
                full = np.full((len(records), len(self.keys[q])), 1e-6)
                for j, cls in enumerate(model.classes_):
                    full[:, int(cls)] = np.maximum(pr[:, j], 1e-6)
                per_cand.append(full / full.sum(axis=1, keepdims=True))
            for i, k in enumerate(keys):
                self.cache.setdefault(k, {})[q] = [c[i] for c in per_cand]
        for i, k in enumerate(keys):
            self.cache[k]["_ntok"] = int(ntok[i])
        self.precompute_seconds = time.time() - t0

    def _fit(self, fam, hp, q, mask):
        if fam == "lr":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(C=hp, max_iter=2000, random_state=self.seed)
            return m.fit(self.X_sparse_train[mask], self.y[q][mask])
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(learning_rate=hp, max_iter=300, random_state=self.seed)
        return m.fit(self.X_dense_train[mask], self.y[q][mask])

    def _features(self, state):
        from scipy import sparse
        n, c, t = flatten(state)
        Xn = self.dv_num.transform([n]); Xc = self.dv_cat.transform([c]); Xt = self.tfidf.transform([t])
        Xn_s = sparse.csr_matrix((Xn.toarray() - self.num_mean) / self.num_std) if Xn.shape[1] else Xn
        xs = sparse.hstack([Xn_s, Xc, Xt]).tocsr()
        xd = np.hstack([Xn.toarray(), self.oe.transform(Xc.toarray()), self.svd.transform(Xt)]) if self.structured else None
        return xs, xd, Xt.nnz

    def _probs(self, model, fam, xs, xd, q):
        p = np.full(len(self.keys[q]), 1e-6)
        pr = model.predict_proba(xs if fam == "lr" else xd)[0]
        for cls, v in zip(model.classes_, pr):
            p[int(cls)] = max(v, 1e-6)
        return p / p.sum()

    def __call__(self, record):
        start = time.perf_counter()
        hit = self.cache.get(self.record_key(record))
        if hit is None:                      # not precomputed (not expected): compute this record alone
            xs, xd, ntok = self._features(record["state"])
        probs, logits = {}, {}
        for q in record["questions"]:
            cands = self.models[q]
            pick = 0 if len(cands) == 1 else self._pick(q)
            if hit is not None:
                p = hit[q][pick]; ntok = hit["_ntok"]
            else:
                fam, hp, model = cands[pick]
                p = self._probs(model, fam, xs, xd, q)
            probs[q] = dict(zip(self.keys[q], p.tolist())); logits[q] = dict(zip(self.keys[q], np.log(p).tolist()))
        return {"probabilities": probs, "logits": logits, "inference_temperature": 1.0,
                "latency_ms": 1000 * (time.perf_counter() - start), "input_tokens": int(ntok)}

    def _pick(self, q):
        return self.choice.get(f"_pick_{q}", 0)


def select_on_dev(pred: ClassicalPredictor, dev_path: str, out_dir: Path) -> None:
    """Pick, per question, the candidate with the best macro-F1 on the dev records; store the choice next to the run."""
    from sklearn.metrics import f1_score
    records = [json.loads(l) for l in open(dev_path) if l.strip()]
    keys = [pred.record_key(r) for r in records]
    choice = {}
    for q in pred.qids:
        y = np.array([pred.keys[q].index(label_key(r["questions"][q])) if q in r["questions"] else -1 for r in records]); mask = y >= 0
        scores = []
        for i in range(len(pred.models[q])):
            yhat = np.array([int(np.argmax(pred.cache[k][q][i])) for k in keys])
            scores.append(f1_score(y[mask], yhat[mask], average="macro"))
        best = int(np.argmax(scores))
        choice[q] = list(pred.models[q][best][:2]); choice[f"_pick_{q}"] = best; choice[f"_dev_macro_f1_{q}"] = [float(s) for s in scores]
    pred.choice = choice
    pred.choice_record = {**choice, "fit_seconds": pred.fit_seconds, "task": pred.task, "seed": pred.seed, "train": pred.train}


def pop_flag(argv, name, default=None):
    if name not in argv:
        return default
    i = argv.index(name); v = argv[i + 1]; del argv[i:i + 2]; return v


def main(argv):
    import kev.benchmark as benchmark
    from kev.model import training_context
    ClassicalPredictor.task = pop_flag(argv, "--task")
    ClassicalPredictor.train = pop_flag(argv, "--train")
    ClassicalPredictor.seed = int(pop_flag(argv, "--seed", "0"))
    ClassicalPredictor.select = pop_flag(argv, "--select")
    max_state = int(pop_flag(argv, "--max_state", "7552"))   # kev.benchmark accepts 384 to 7552; the classical features do not truncate
    benchmark.CONTEXT = {**training_context(max_state), "truncate": False}
    out = Path(argv[argv.index("--out") + 1]); data = argv[argv.index("--data") + 1]
    ClassicalPredictor.data = data
    pred = ClassicalPredictor(None, "cpu")
    if ClassicalPredictor.select is None:      # dev pass: choose family/C on these dev records first, then score them
        select_on_dev(pred, data, out)
    else:
        pred.choice_record = json.loads(Path(ClassicalPredictor.select, "classical_choice.json").read_text())
    if "--run" not in argv:
        argv = ["--run", "classical"] + argv     # kev.benchmark wants a run name; the predictor is swapped below
    benchmark.LocalPredictor = lambda *a, **k: pred
    sys.argv = [sys.argv[0]] + argv
    benchmark.main()                              # creates --out itself (refuses an existing directory)
    (out / "classical_choice.json").write_text(json.dumps({**pred.choice_record, "precompute_seconds": pred.precompute_seconds}, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
