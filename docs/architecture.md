# Architecture

The v1 pipeline, from raw CSVs to a served prediction. `make all` runs everything
up to the test report; `docker compose up` serves the result.

```mermaid
flowchart TD
    raw["Olist CSVs<br/>make data, checksums verified"]
    subgraph warehouse["DuckDB warehouse, built by dbt"]
        staging["7 staging models<br/>types, names, column tests"]
        history["As-of history<br/>seller and route, known before t0"]
        mart["Feature table<br/>25 features, one row per order"]
    end
    checks["Leakage checks<br/>dbt tests + pandas brute force"]
    selection["Model selection<br/>3 folds + validation<br/>baselines, XGBoost search, MLflow"]
    calibration["Calibration backtest<br/>each option replayed forward in time"]
    bundle["Model bundle<br/>preprocessing, model, calibrator, version"]
    protocol["Test protocol<br/>committed before scoring"]
    report["Test report, scored once<br/>evaluation.md, model card"]
    api["FastAPI in Docker<br/>predict, explain, model-info"]
    log["Prediction log<br/>groundwork for v2 monitoring"]

    raw --> staging --> history --> mart
    mart --> checks
    mart --> selection --> calibration --> bundle
    bundle --> protocol --> report
    bundle --> api
    mart --> api
    api --> log
```

| Stage | Code | Output |
| --- | --- | --- |
| Download | `src/opspilot/data/` | `data/raw/` (gitignored), `data/sample/` for CI |
| Warehouse | `warehouse/` | `warehouse/opspilot.duckdb` |
| Selection | `src/opspilot/models/` | MLflow runs, `artifacts/`, `reports/selection_tables.md` |
| Evaluation | `src/opspilot/models/evaluate.py` | `reports/evaluation.md`, `reports/figures/` |
| Serving | `src/opspilot/api/`, `docker/` | the API on port 8000, the MLflow UI on port 5001 |

CI runs lint, the warehouse and its tests, all Python tests, the whole pipeline on
the sample with a tiny search, and the API image build.
