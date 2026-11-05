"""Cohort SQL for the internal tasks, FMX Postgres conventions: interned codes via fmx_read_code, tstzrange dates via
lower(), camelCase columns quoted, references 'Encounter/<id>'. count_* queries return (yr, step, n) aggregates only;
extract_* queries return rows and run inside the hospital network only."""
from __future__ import annotations

import re

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ICD10GM = "http://fhir.de/CodeSystem/bfarm/icd-10-gm"
ALPHA_ID = "http://fhir.de/CodeSystem/bfarm/alpha-id"
ICDO3 = "http://terminology.hl7.org/CodeSystem/icd-o-3"
KDL = "http://dvmd.de/fhir/CodeSystem/kdl"
ISP_DOC = "https://uk-essen.de/fhir/isp/dokument"
LETTER_KDL = "AD010103"


def _check(start: str, end: str) -> None:
    if not (_DATE.match(start) and _DATE.match(end)):
        raise ValueError("start and end must be ISO dates YYYY-MM-DD")


def imp_cases_cte(start: str, end: str, name: str = "imp") -> str:
    _check(start, end)
    return f"""{name} AS (
  SELECT e._id AS _rid, e.id AS case_id, es._reference_id AS patient_rid, split_part(es.reference, '/', 2) AS patient_id,
         lower(ep.start) AS admit, lower(ep."end") AS discharge, EXTRACT(YEAR FROM lower(ep.start))::int AS yr,
         (SELECT stc.display FROM "encounter_serviceType" est JOIN "encounter_serviceType_coding" stc ON stc._parent = est._id
            WHERE est._resource = e._id LIMIT 1) AS department
  FROM encounter e
  JOIN encounter_period ep ON ep._resource = e._id
  JOIN encounter_class ec ON ec._resource = e._id AND ec.code = fmx_read_code('IMP')
  JOIN encounter_type_coding etc ON etc._resource = e._id AND etc.code = fmx_read_code('einrichtungskontakt')
  LEFT JOIN encounter_subject es ON es._resource = e._id
  WHERE e.status = fmx_read_code('finished')
    AND ep.start && tstzrange('{start}', '{end}', '[)')
    AND lower(ep.start) >= '{start}' AND lower(ep.start) < '{end}'
)"""


def _imp_all_cte(start: str) -> str:
    """Every finished IMP case admitted from `start` on, unbounded above: the I4 readmission lookup must see admissions
    after the extraction window (the extract runs per 3-month window); a next admission is always after the discharge,
    so nothing before `start` can qualify. `&&` on the tstzrange uses the GiST index, `lower()` makes the bound exact."""
    _check(start, start)
    return f"""imp_all AS (
  SELECT e._id AS _rid, es._reference_id AS patient_rid, lower(ep.start) AS admit
  FROM encounter e
  JOIN encounter_period ep ON ep._resource = e._id
  JOIN encounter_class ec ON ec._resource = e._id AND ec.code = fmx_read_code('IMP')
  JOIN encounter_type_coding etc ON etc._resource = e._id AND etc.code = fmx_read_code('einrichtungskontakt')
  LEFT JOIN encounter_subject es ON es._resource = e._id
  WHERE e.status = fmx_read_code('finished')
    AND ep.start && tstzrange('{start}', NULL, '[)')
    AND lower(ep.start) >= '{start}'
)"""


def _auf_cte() -> str:
    return f"""auf AS (
  SELECT imp.case_id AS case_id,
         array_agg(fmx_read_code(ccc.code)) FILTER (WHERE ccc.system = fmx_read_code('{ICD10GM}')) AS auf_codes,
         array_agg(ccc.display) FILTER (WHERE ccc.system = fmx_read_code('{ICD10GM}')) AS auf_texts,
         bool_and(ccc.system = fmx_read_code('{ALPHA_ID}')) AS auf_alpha_only
  FROM imp
  JOIN condition_encounter ce ON ce._reference_type = 'Encounter' AND ce._reference_id = imp._rid
  JOIN condition c ON c._id = ce._resource
  JOIN condition_category_coding cat ON cat._resource = c._id AND cat.code = fmx_read_code('AUF')
  JOIN condition_code_coding ccc ON ccc._resource = c._id
  GROUP BY 1
)"""


def count_i1(start: str, end: str) -> str:
    return f"""WITH {imp_cases_cte(start, end)}, {_auf_cte()}
SELECT imp.yr, step, COUNT(*) AS n FROM imp LEFT JOIN auf ON auf.case_id = imp.case_id
CROSS JOIN LATERAL (VALUES
  ('imp_cases'),
  ('status_finished'),
  (CASE WHEN imp.discharge IS NULL THEN 'open' END),
  (CASE WHEN imp.discharge < imp.admit THEN 'invalid_period' END),
  (CASE WHEN auf.auf_codes IS NOT NULL THEN 'with_auf_icd' END),
  (CASE WHEN auf.auf_alpha_only THEN 'auf_alpha_only' END)) v(step)
WHERE step IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2"""


def extract_i1(start: str, end: str) -> str:
    return f"""WITH {imp_cases_cte(start, end)}, {_auf_cte()}
SELECT imp.case_id AS case_id, imp.patient_id AS patient_id, imp.admit AS admit, imp.discharge AS discharge, imp.yr AS year,
       imp.department AS department,
       EXTRACT(YEAR FROM age(imp.admit, lower(p."birthDate")))::int AS age, fmx_read_code(p.gender) AS sex,
       auf.auf_codes AS auf_codes, auf.auf_texts AS auf_texts, COALESCE(auf.auf_alpha_only, false) AS auf_alpha_only
FROM imp
LEFT JOIN auf ON auf.case_id = imp.case_id
LEFT JOIN patient p ON p._id = imp.patient_rid"""


def extract_i1_labs(start: str, end: str) -> str:
    return f"""WITH {imp_cases_cte(start, end)},
labs AS (
  SELECT imp.case_id AS case_id, oc.display AS name, fmx_read_code(oc.code) AS code, ovq.value AS value,
         ovq.unit AS unit, lower(o."effectiveDateTime") AS taken,
         ROW_NUMBER() OVER (PARTITION BY imp.case_id ORDER BY lower(o."effectiveDateTime"), o._id) AS rn
  FROM imp
  JOIN observation_encounter oe ON oe._reference_type = 'Encounter' AND oe._reference_id = imp._rid
  JOIN observation o ON o._id = oe._resource
  JOIN observation_category_coding occ ON occ._resource = o._id AND occ.code = fmx_read_code('laboratory')
  JOIN observation_code_coding oc ON oc._resource = o._id
  JOIN "observation_valueQuantity" ovq ON ovq._resource = o._id
  WHERE lower(o."effectiveDateTime") >= imp.admit AND lower(o."effectiveDateTime") < imp.admit + INTERVAL '24 hours'
)
SELECT case_id AS case_id, name AS name, code AS code, value AS value, unit AS unit, taken AS taken FROM labs WHERE rn <= 30"""


def extract_i1_history(start: str, end: str) -> str:
    return f"""WITH {imp_cases_cte(start, end)}
SELECT imp.case_id AS case_id,
       array_agg(DISTINCT fmx_read_code(ccc.code)) AS prior_codes, array_agg(DISTINCT ccc.display) AS prior_texts
FROM imp
JOIN encounter_subject es2 ON es2._reference_type = 'Patient' AND es2._reference_id = imp.patient_rid
JOIN encounter e2 ON e2._id = es2._resource AND e2._id <> imp._rid AND e2.status = fmx_read_code('finished')
JOIN encounter_period ep2 ON ep2._resource = e2._id AND lower(ep2."end") < imp.admit
JOIN condition_encounter ce ON ce._reference_type = 'Encounter' AND ce._reference_id = e2._id
JOIN condition c ON c._id = ce._resource
JOIN condition_category_coding cat ON cat._resource = c._id AND cat.code IN (fmx_read_code('ENT'), fmx_read_code('drg'))
JOIN condition_code_coding ccc ON ccc._resource = c._id AND ccc.system = fmx_read_code('{ICD10GM}')
GROUP BY 1"""


def _drg_cte() -> str:
    return f"""drg AS MATERIALIZED (
  SELECT imp.case_id AS case_id, COUNT(DISTINCT c._id) AS n_drg,
         MAX(fmx_read_code(ccc.code)) AS drg_code, MAX(ccc.version) AS icd_version
  FROM imp
  JOIN condition_encounter ce ON ce._reference_type = 'Encounter' AND ce._reference_id = imp._rid
  JOIN condition c ON c._id = ce._resource
  JOIN condition_category_coding cat ON cat._resource = c._id AND cat.code = fmx_read_code('drg')
  JOIN condition_code_coding ccc ON ccc._resource = c._id AND ccc.system = fmx_read_code('{ICD10GM}')
  GROUP BY 1
)"""


def _letters_cte() -> str:
    return f"""letters AS MATERIALIZED (
  SELECT imp.case_id AS case_id, d.id AS doc_id, d.date AS doc_date, a.url AS pdf_url
  FROM imp
  JOIN documentreference_context_encounter cenc ON cenc._reference_type = 'Encounter' AND cenc._reference_id = imp._rid
  JOIN documentreference_context ctx ON ctx._id = cenc._parent
  JOIN documentreference d ON d._id = ctx._resource
  JOIN documentreference_type_coding tc ON tc._resource = d._id AND tc.system = fmx_read_code('{KDL}') AND tc.code = fmx_read_code('{LETTER_KDL}')
  JOIN documentreference_content c ON c._resource = d._id
  JOIN documentreference_content_attachment a ON a._parent = c._id AND a."contentType" = fmx_read_code('application/pdf')
  WHERE d."docStatus" = fmx_read_code('final')
)"""


def count_i2(start: str, end: str) -> str:
    return f"""WITH {imp_cases_cte(start, end)}, {_drg_cte()}, {_letters_cte()},
per_case AS (SELECT imp.yr, imp.case_id, d.n_drg, COUNT(l.doc_id) AS n_letters
             FROM imp LEFT JOIN drg d ON d.case_id = imp.case_id LEFT JOIN letters l ON l.case_id = imp.case_id GROUP BY 1, 2, 3)
SELECT yr, step, COUNT(*) AS n FROM per_case
CROSS JOIN LATERAL (VALUES ('imp_cases'), (CASE WHEN n_drg = 1 THEN 'single_drg_icd' END), (CASE WHEN n_drg > 1 THEN 'multi_drg' END),
  (CASE WHEN n_drg IS NULL THEN 'no_drg_icd' END), (CASE WHEN n_drg = 1 AND n_letters >= 1 THEN 'eligible_with_letter' END),
  (CASE WHEN n_drg = 1 AND n_letters > 1 THEN 'eligible_multi_letter' END)) v(step)
WHERE step IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2"""


def extract_i2(start: str, end: str) -> str:
    return f"""WITH {imp_cases_cte(start, end)}, {_drg_cte()}, {_letters_cte()}
SELECT imp.case_id AS case_id, imp.patient_id AS patient_id, imp.yr AS year, d.drg_code AS drg_code, d.icd_version AS icd_version,
       l.doc_id AS doc_id, l.doc_date AS doc_date, l.pdf_url AS pdf_url
FROM imp JOIN drg d ON d.case_id = imp.case_id AND d.n_drg = 1
JOIN letters l ON l.case_id = imp.case_id"""


def _isp_cte(start: str, end: str) -> str:
    _check(start, end)
    return f"""isp AS (
  SELECT dr._id, dr.id AS report_id, split_part(ds.reference, '/', 2) AS patient_id,
         lower(dr."effectiveDateTime") AS issued, EXTRACT(YEAR FROM lower(dr."effectiveDateTime"))::int AS yr
  FROM diagnosticreport dr
  JOIN diagnosticreport_identifier di ON di._resource = dr._id AND di.system = fmx_read_code('{ISP_DOC}')
  LEFT JOIN diagnosticreport_subject ds ON ds._resource = dr._id
  WHERE lower(dr."effectiveDateTime") >= '{start}' AND lower(dr."effectiveDateTime") < '{end}'
), codes AS (
  SELECT dcc._resource AS _id, array_agg(fmx_read_code(cc.code)) AS icdo3_codes
  FROM "diagnosticreport_conclusionCode" dcc
  JOIN "diagnosticreport_conclusionCode_coding" cc ON cc._parent = dcc._id AND cc.system = fmx_read_code('{ICDO3}')
  GROUP BY 1
), md AS (
  SELECT pf._resource AS _id, MAX(pf.size) AS md_size, COUNT(*) AS n_md
  FROM "diagnosticreport_presentedForm" pf WHERE pf."contentType" = fmx_read_code('text/markdown') GROUP BY 1
)"""


def count_i3(start: str, end: str) -> str:
    return f"""WITH {_isp_cte(start, end)}
SELECT isp.yr, step, COUNT(*) AS n FROM isp LEFT JOIN codes ON codes._id = isp._id LEFT JOIN md ON md._id = isp._id
CROSS JOIN LATERAL (VALUES ('isp_reports'), (CASE WHEN codes.icdo3_codes IS NOT NULL THEN 'with_icdo3' END),
  (CASE WHEN codes.icdo3_codes IS NOT NULL AND md.n_md >= 1 THEN 'with_icdo3_and_markdown' END),
  (CASE WHEN array_length(codes.icdo3_codes, 1) > 1 THEN 'multi_code' END)) v(step)
WHERE step IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2"""


def extract_i3(start: str, end: str) -> str:
    return f"""WITH {_isp_cte(start, end)}
SELECT isp.report_id AS report_id, isp.patient_id AS patient_id, isp.yr AS year, isp.issued AS issued,
       codes.icdo3_codes AS icdo3_codes,
       convert_from(decode(pf.data, 'base64'), 'UTF8') AS markdown
FROM isp JOIN codes ON codes._id = isp._id
JOIN "diagnosticreport_presentedForm" pf ON pf._resource = isp._id AND pf."contentType" = fmx_read_code('text/markdown')"""


def count_i4(start: str, end: str) -> str:
    return f"""WITH {imp_cases_cte(start, end)}, {_imp_all_cte(start)},
nxt AS (SELECT a.case_id, MIN(b.admit) AS next_admit FROM imp a JOIN imp_all b ON b.patient_rid = a.patient_rid AND b.admit > a.discharge GROUP BY 1)
SELECT imp.yr, step, COUNT(*) AS n FROM imp LEFT JOIN nxt ON nxt.case_id = imp.case_id
CROSS JOIN LATERAL (VALUES ('imp_cases'), (CASE WHEN nxt.next_admit IS NOT NULL AND nxt.next_admit < imp.discharge + INTERVAL '30 days' THEN 'readmitted_30d' END)) v(step)
WHERE step IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2"""


def extract_i4(start: str, end: str) -> str:
    return f"""WITH {imp_cases_cte(start, end)}, {_imp_all_cte(start)},
nxt AS (SELECT a.case_id, MIN(b.admit) AS next_admit FROM imp a JOIN imp_all b ON b.patient_rid = a.patient_rid AND b.admit > a.discharge GROUP BY 1)
SELECT imp.case_id AS case_id, imp.patient_id AS patient_id, imp.discharge AS discharge, nxt.next_admit AS next_admit
FROM imp LEFT JOIN nxt ON nxt.case_id = imp.case_id"""
