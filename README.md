# OpsPilot AI

Predicts which e-commerce orders will arrive late, using only what was known
when the order was approved.

> **Status:** work in progress. Week 1 of 9 is done (setup, EDA, design
> decisions). `v1.0.0` is planned for 14 December 2026. Results, architecture
> and limitations land here as each part is built.

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

The decisions that shape everything else, with the numbers behind them
([EDA notebook](notebooks/01_eda.ipynb)):

1. [Target definition](docs/decisions/0001-target-definition.md): late means delivered after the promised calendar date; 6.77% of delivered orders.
2. [Prediction time](docs/decisions/0002-prediction-time.md): features must be known when the order is approved.
3. [Censoring cutoff](docs/decisions/0003-censoring-cutoff.md): orders purchased after 2018-07-15 are dropped, because their slow deliveries are missing from the data.
4. [Chronological split](docs/decisions/0004-chronological-split.md): train on 2017, validate on Q1 2018, test on April to mid-July 2018.
5. [Leakage rules](docs/decisions/0005-leakage-rules.md): nothing recorded after approval, and seller history only from orders already delivered.

## License

Code: [MIT](LICENSE). Data: CC BY-NC-SA 4.0, owned by Olist; see
[data/README.md](data/README.md).
