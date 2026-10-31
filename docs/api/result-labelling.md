# Result labelling rules (MSP-32)

Owner: Nishitha. All fixed wording lives in `backend/core/labels.py`; code builds result blocks from its
helpers and never writes its own sentences. `tests/unit/test_result_schema.py` checks every tier.

* **Every result block** (a dict with a value and its explanation) carries `unit`, `limitation` and
  `model_version`, all non-empty (NFR-007, NFR-008). The test walks one real payload per tier (ML, DL, QML) and fails
  on any block that has some of the three but not all.
* **Required blocks:** ML `solubility`, `tox21`, `alerts`, `rules`, `domain`; DL `activity`, `baseline`;
  QML `electronic_stability`.
* **Tox21** values carry `label: "assay_signal"` (BR-006) and the limitation names what Tox21 does not cover
  (hERG, hepatotoxicity, mutagenicity; NFR-009).
* **No forbidden claims** (NFR-011): no sentence in `labels.py` and no text in any payload uses the whole words
  safe, non-toxic, nontoxic, harmless, effective, approved, cure, cured (`labels.forbidden_claims_in`).
* **QML:** while the stub backend is used the block is marked three ways (`is_stub`, `backend.is_stub`, a limitation
  starting `STUB:`); `labels.qml_labels(is_stub, model_version)` returns the stub or the real-backend wording.
* **Disclaimer:** the job status body (`GET /api/jobs/<id>`) has a `disclaimer` field with `labels.DISCLAIMER`.
  The results screen must show it with the results.
