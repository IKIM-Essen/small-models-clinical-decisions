"""Context strategies for long states: applied identically to Kev and to the
LoRA-SFT baseline before any model sees the text. (a) raise the limit is a training flag, not a transform;
(b) section-aware truncation; (c) chunk-and-aggregate."""
from __future__ import annotations


def _ids(text: str, tok) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def truncate(text: str, max_tokens: int, tok) -> tuple[str, bool]:
    ids = _ids(text, tok)
    if len(ids) <= max_tokens:
        return text, False
    return tok.decode(ids[:max_tokens]), True


def _norm_heading(line: str) -> str:
    """Strip markdown (#, *, _), a trailing colon and case, so `## Diagnosen:` == `**Diagnosen**` == `diagnosen`."""
    return line.strip().strip("#*_ ").rstrip(":").strip().casefold()


def _sections(text: str, named: set[str] = frozenset()) -> list[tuple[str, str]]:
    """Split into (heading, body). A markdown heading line (`#`) is always a boundary. With `named` given, other
    lines are headings only when their normalised form is one of `named` (any length); a one-word capitalised line
    inside a named section stays body text. Without `named`, one capitalised word without trailing punctuation is a
    heading. Returns headings normalised."""
    out, head, buf = [], "", []
    for line in text.splitlines():
        s = line.strip()
        n = _norm_heading(s)
        one_word = bool(n) and len(n) <= 40 and not s.endswith((".", ",", ";")) and " " not in n and s.lstrip("#*_ ")[:1].isupper()
        markdown = s.startswith("#") and bool(n)
        if n and (markdown or ((n in named) if named else one_word)):
            if buf or head:
                out.append((head, "\n".join(buf)))
            head, buf = n, []
        else:
            buf.append(line)
    out.append((head, "\n".join(buf)))
    return out


def section_first(text: str, headings: list[str], max_tokens: int, tok) -> tuple[str, bool]:
    wanted = [_norm_heading(h) for h in headings]
    secs = _sections(text, set(wanted))
    if not any(h in wanted for h, _ in secs):
        return truncate(text, max_tokens, tok)
    front = [s for h in wanted for s in secs if s[0] == h]
    rest = [s for s in secs if s[0] not in wanted]
    ordered = "\n".join(f"{h.capitalize()}\n{b}".strip() if h else b.strip() for h, b in front + rest)
    return truncate(ordered, max_tokens, tok)


def chunks(text: str, max_tokens: int, tok, overlap: int = 32) -> list[str]:
    ids = _ids(text, tok)
    if len(ids) <= max_tokens:
        return [text]
    step = max_tokens - overlap
    return [tok.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), step) if i < len(ids)]
