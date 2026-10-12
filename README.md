# OpsPilot AI

Predicts which e-commerce orders will arrive late, using only what was known
when the order was approved.

> **Status:** work in progress. Week 1 of 9 (setup and EDA). `v1.0.0` is planned
> for 14 December 2026. Results, architecture and limitations land here as each
> part is built.

## What v1 will be

A reproducible pipeline on the public [Olist](data/README.md) dataset (about 100k
orders): a DuckDB + dbt warehouse, leakage-tested features, three honest
baselines against a calibrated XGBoost model, MLflow tracking, and a Dockerized
FastAPI service that returns a late-delivery probability with its top reasons.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv installs it for you).
On macOS, XGBoost also needs OpenMP: `brew install libomp`.

```bash
make setup   # install dependencies and git hooks
make data    # download the Olist CSVs into data/raw and verify checksums
make test
```

`make help` lists every target; the later pipeline stages are added week by week.

## Design decisions

The decisions that shape everything else are recorded in
[docs/decisions/](docs/decisions/).

## License

Code: [MIT](LICENSE). Data: CC BY-NC-SA 4.0, owned by Olist; see
[data/README.md](data/README.md).
