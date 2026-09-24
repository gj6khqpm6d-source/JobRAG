# JobRAG evaluation set

`questions.jsonl` is the initial 25-question evaluation set. It is intentionally created before running the benchmark so retrieval changes can be compared against the same questions.

Each record contains:

- `category`: `skills`, `experience`, `comparison`, `filters`, or `insufficient_evidence`
- `filters`: optional source, location, title, or other retrieval constraints
- `expected_terms`: terms to support manual answer review
- `relevant_job_ids`: legacy inline labels; the local annotation workbench now stores reviewed labels separately
- `should_refuse`: whether the knowledge base should state that evidence is insufficient

Use `http://127.0.0.1:8000/eval` to create human relevance judgments. They are written to `annotations.json` with grades 0 (irrelevant), 1 (partially relevant), and 2 (highly relevant), plus selected evidence chunks and answer key points.

`splits.json` keeps 15 development questions, 5 untouched test questions, and 5 refusal questions separate. The tuning command only uses reviewed development annotations and refuses to choose a best configuration until at least five development questions have positive relevance labels:

```bash
.venv/bin/python -m app.eval.tuning
```

The report is written to `reports/retrieval-tuning.json`. Test questions remain unevaluated during tuning to avoid selecting parameters against the holdout set.

## Local reranker experiment

`app/eval/reranker_experiment.py` evaluates the local `BAAI/bge-reranker-v2-m3` cross-encoder on pooled Keyword, Vector, and Hybrid candidates. It calibrates job-level score thresholds only on development annotations and reports ranking metrics, candidate-pool coverage, classification accuracy, and a disagreement queue. The experiment does not change the production retrieval configuration:

```bash
RERANKER_MAX_LENGTH=256 .venv/bin/python -m app.eval.reranker_experiment \
  --pool-k 10 \
  --output data/eval/reports/reranker-experiment.json
```

The current report shows the reranker as an offline research result rather than a production component: it is slower on CPU and must be audited before its scores are used as silver labels.

## End-to-end quality gate

`app/eval/quality.py` runs the production quality gate across five layers:

- evaluation-dataset health and human-calibration coverage;
- job-level Recall@8, MRR, graded NDCG@8, and metadata-filter accuracy;
- job deduplication, skill-denominator, and percentage invariants;
- citation validity and refusal accuracy;
- active-job freshness, inactive-index isolation, and embedding coverage.

Thresholds are versioned in `quality-gates.json`; the latest atomic report is written to `reports/quality-latest.json`. The default run uses a deterministic local answer provider so it validates the pipeline without DeepSeek or judge-model cost. Human annotations remain the calibration source for retrieval relevance.

Legacy annotations contain incomplete judgments, so unreviewed jobs are not treated as negative examples. Recall@8 measures known-positive coverage and condensed graded NDCG@8 measures the ordering of known graded positives. Ranking changes are selected on the development split; only the independent test split controls the retrieval release gate. The annotation UI now stores “未判断” separately from an explicit relevance grade of 0.

Run it from the admin page at `http://127.0.0.1:8000/eval`, or locally:

```bash
.venv/bin/python -c "from app.eval.quality import run_quality_evaluation; run_quality_evaluation()"
```

## Small-sample response-quality evaluation

`app.eval.response_quality` evaluates two locked holdout answers and one insufficient-evidence answer with the configured DeepSeek model as both generator and judge. It reuses versioned answer-cache entries and sends all three cases to one judge request. A cold run therefore uses at most four model calls; cached answers reduce that count. To bound prompt and completion tokens, the judge receives only cited evidence (at most 2,400 characters per case), checks at most two claims and five key points per answer, and returns compact JSON. The report includes claim-level evidence checks, groundedness, answer relevance, completeness, refusal judgment, latency, and returned token usage. Since the same model generates and judges, treat semantic scores as diagnostic until a few cases are independently reviewed.

```bash
.venv/bin/python -m app.eval.response_quality
```

The default sample is capped at three questions and cannot read development questions. The report is written to `reports/response-quality-latest.json`. Key-point coverage uses human answer points when available, otherwise it labels `expected_terms` as a weak proxy.

A failed gate is an actionable release blocker, not a runtime failure. The report preserves the failed question rows so retrieval or refusal behavior can be corrected and rerun against the same benchmark.
