import re

import pytest

from kevmed.fmx import sql

FORBIDDEN = re.compile(r"\b(text|display|presentedForm|url|markdown|data)\b", re.I)


@pytest.mark.parametrize("fn", [sql.count_i1, sql.count_i2, sql.count_i3, sql.count_i4])
def test_count_queries_are_aggregates(fn):
    q = fn("2021-01-01", "2027-01-01")
    outer_select = q.strip().rsplit("SELECT", 1)[-1].split("FROM")[0]
    assert "COUNT(" in outer_select.upper()
    assert not FORBIDDEN.search(outer_select)
    assert "GROUP BY" in q.upper()


@pytest.mark.parametrize("fn,cols", [
    (sql.extract_i1, ["case_id", "patient_id", "admit", "discharge", "year", "department", "age", "sex", "auf_codes", "auf_texts"]),
    (sql.extract_i2, ["case_id", "patient_id", "year", "drg_code", "icd_version", "doc_id", "doc_date", "pdf_url"]),
    (sql.extract_i3, ["report_id", "patient_id", "year", "issued", "icdo3_codes", "markdown"]),
    (sql.extract_i4, ["case_id", "patient_id", "discharge", "next_admit"]),
])
def test_extract_queries_name_their_columns(fn, cols):
    q = fn("2023-01-01", "2024-01-01")
    for c in cols:
        assert re.search(rf"\bAS {c}\b", q), c
    assert "'2023-01-01'" in q and "'2024-01-01'" in q


def test_year_bounds_must_be_iso_dates():
    with pytest.raises(ValueError):
        sql.count_i1("2021", "2027-01-01")
    with pytest.raises(ValueError):
        sql.extract_i3("2019-01-01", "2027-01-01; DROP TABLE x")


def test_db_count_refuses_row_queries(monkeypatch):
    from kevmed.fmx import db
    monkeypatch.setenv("FMX_DSN", "postgresql://u:p@h/db")
    with pytest.raises(ValueError):
        db.count("SELECT text FROM documentreference LIMIT 5")


def test_db_dsn_required(monkeypatch):
    from kevmed.fmx import db
    monkeypatch.delenv("FMX_DSN", raising=False)
    with pytest.raises(RuntimeError):
        db.dsn()


def test_cohort_keeps_only_finished_cases():
    for fn in (sql.count_i1, sql.extract_i1, sql.count_i2, sql.extract_i2, sql.count_i4, sql.extract_i4):
        q = fn("2021-01-01", "2027-01-01")
        assert "e.status = fmx_read_code('finished')" in q, fn.__name__
    assert "'status_finished'" in sql.count_i1("2021-01-01", "2027-01-01")


def test_extract_i1_uses_inline_birthdate_and_plain_unit():
    q = sql.extract_i1("2021-01-01", "2027-01-01")
    assert 'patient_birthDate' not in q and 'lower(p."birthDate")' in q
    labs = sql.extract_i1_labs("2021-01-01", "2027-01-01")
    assert "ovq.unit AS unit" in labs and "fmx_read_code(ovq.unit)" not in labs
    assert "ROW_NUMBER() OVER (PARTITION BY" in labs


def test_reference_joins_use_the_internal_reference_columns():
    """FMX indexes `(_reference_type, _reference_id)` (and `_reference`), not the `reference` string: the string form
    only matches the trigram index and the 2026-09-28 extract windows fell back to hashing every AUF condition."""
    for fn in (sql.count_i1, sql.extract_i1, sql.extract_i1_labs, sql.extract_i1_history, sql.count_i2, sql.extract_i2,
               sql.count_i4, sql.extract_i4):
        q = fn("2021-01-01", "2021-04-01")
        assert "reference = 'Encounter/' ||" not in q and "reference = 'Patient/' ||" not in q, fn.__name__
        assert "split_part(oe.reference" not in q and "split_part(es2.reference" not in q
    labs = sql.extract_i1_labs("2021-01-01", "2021-04-01")
    assert "oe._reference_type = 'Encounter' AND oe._reference_id = imp._rid" in labs
    hist = sql.extract_i1_history("2021-01-01", "2021-04-01")
    assert "es2._reference_type = 'Patient' AND es2._reference_id = imp.patient_rid" in hist
    assert "ce._reference_type = 'Encounter' AND ce._reference_id = e2._id" in hist
    assert "p._id = imp.patient_rid" in sql.extract_i1("2021-01-01", "2021-04-01")


def test_cohort_window_uses_the_period_gist_index():
    q = sql.extract_i1("2021-01-01", "2021-04-01")
    imp = q[q.index("imp AS ("):q.index("\n)")]
    assert "ep.start && tstzrange('2021-01-01', '2021-04-01', '[)')" in imp
    assert "lower(ep.start) >= '2021-01-01' AND lower(ep.start) < '2021-04-01'" in imp
    q4 = sql.extract_i4("2021-01-01", "2021-04-01")
    imp_all = q4[q4.index("imp_all AS"):q4.index("nxt AS")]
    assert "ep.start && tstzrange('2021-01-01', NULL, '[)')" in imp_all and "2021-04-01" not in imp_all


def test_db_count_accepts_registered_statements_only(monkeypatch):
    from kevmed.fmx import db
    monkeypatch.setenv("FMX_DSN", "postgresql://u:p@h/db")
    with pytest.raises(ValueError):
        db.count("SELECT d.description, COUNT(*) FROM documentreference d GROUP BY 1")
    with pytest.raises(ValueError):
        db.count(sql.extract_i3("2019-01-01", "2020-01-01"))
    assert db.registered(sql.count_i3("2019-01-01", "2020-01-01"))
    assert not db.registered("SELECT COUNT(*) FROM patient GROUP BY 1")


def test_i4_next_admission_lookup_is_not_bounded_by_the_extraction_window():
    """Extract runs per 3-month window (fmx_extract --chunk-months); the readmission lookup must see later windows."""
    for fn in (sql.count_i4, sql.extract_i4):
        q = fn("2021-01-01", "2021-04-01")
        nxt = q[q.index("nxt AS"):q.index("GROUP BY 1)")]
        assert "JOIN imp_all b" in nxt and "JOIN imp b" not in nxt
        imp_all = q[q.index("imp_all AS"):q.index("nxt AS")]
        assert "2021-04-01" not in imp_all and "fmx_read_code('finished')" in imp_all


def test_condition_and_letter_ctes_are_scoped_to_the_cohort():
    """The AUF, DRG and letters CTEs start from `imp` (index lookup by 'Encounter/<id>'), not from every condition or
    document in FMX: the extract runs once per 3-month window, an unscoped aggregate would be recomputed every window."""
    for fn in (sql.count_i1, sql.extract_i1, sql.count_i2, sql.extract_i2):
        q = fn("2021-01-01", "2021-04-01")
        for name in ("auf AS", "drg AS", "letters AS"):
            if name in q:
                cte = q[q.index(name):]
                cte = cte[:cte.index("\n)")]
                assert "FROM imp\n" in cte and "_reference_type = 'Encounter' AND" in cte and "_reference_id = imp._rid" in cte, name


def test_i2_ctes_are_materialized():
    """EXPLAIN ANALYZE on prod (2026-09-28): inlined, the letters subquery was re-executed once per DRG row
    (959 loops for a one-week window, 207k letter type codings each), 228 s for a week. MATERIALIZED evaluates
    each CTE once per window."""
    for fn in (sql.count_i2, sql.extract_i2):
        q = fn("2023-01-01", "2023-04-01")
        assert "drg AS MATERIALIZED (" in q and "letters AS MATERIALIZED (" in q
