# OpsPilot AI

Predicts which e-commerce orders will arrive late, using only what was known
when the order was approved.

> **Status:** work in progress. Weeks 1-8 of 9 are done: the warehouse, a
> leakage-tested feature table, baselines, a tuned and calibrated XGBoost chosen
> on validation data only, a single pre-registered scoring of the test set, and
> a Dockerized prediction API. Release polish (Week 9) is next. `v1.0.0` is
> planned for 14 December 2026.

## Results

Scored once on the test set (April to mid-July 2018, 21,949 orders, 4.05% late),
after the model and the evaluation plan were committed in
[reports/test_protocol.md](reports/test_protocol.md).

- **Ranking works.** Checking the riskiest 10% of orders finds 28.6% of all late
  orders, against 15.2% for ranking by the route's late rate and 10% at random
  (PR-AUC 0.094 against a random 0.041).
- **Logistic regression did better on test** (PR-AUC 0.116, 35.3% of late
  orders in the riskiest 10%), although XGBoost had led on validation. The model
  was not switched after seeing this, and it remains the served model. Retraining
  XGBoost monthly improved it in every test month.
- **Calibration drifted.** Probabilities fitted on periods running up to 13%
  late came out about four times too high in a 4% period. Ranking held; the
  level did not.

Details: [evaluation](reports/evaluation.md), [model card](reports/model_card.md),
[model selection](reports/model_selection.md),
[error analysis](notebooks/02_error_analysis.ipynb).

## Use the API

After `make features train`:

```bash
docker compose up --build
```

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"order_id": "7af07887f8840085ae4b65cb52f4af29"}'
```

The response has the probability, a risk band, the model version and the top
three reasons. Docs are at <http://localhost:8000/docs>, and the MLflow UI is at
<http://localhost:5001>. `/explain` returns every feature's contribution,
`/model-info` describes the model, and each prediction is appended to a log table
for future monitoring.

- **Read the band more than the percentage.** The bands were set on validation.
  On test they still sorted orders well (low 1.8% late, medium 6.0%, high 12.0%),
  but the probabilities themselves ran high ([why](docs/decisions/0006-calibration-under-shift.md)).
- **This is batch scoring, not a live service.** `order_id` looks up features
  already built in the warehouse, so it covers historical orders only. Sending
  raw `features` works for any order, but the caller must supply history values
  such as the route's late rate, which a real deployment would compute at request
  time.

## What v1 will be

A reproducible pipeline on the public [Olist](data/README.md) dataset (about 100k
orders): a DuckDB + dbt warehouse, leakage-tested features, three honest
baselines against a calibrated XGBoost model, MLflow tracking, and a Dockerized
FastAPI service that returns a late-delivery probability with its top reasons.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv installs it for you).
On macOS, XGBoost also needs OpenMP: `brew install libomp`.

```bash
make setup      # install dependencies and git hooks
make data       # download the Olist CSVs into data/raw and verify checksums
make dbt        # build and test the staging layer in DuckDB
make features   # build and test the feature table, including the leakage check
make train      # baselines, then tune XGBoost and save the calibrated model bundle
make mlflow     # browse the runs at http://127.0.0.1:5001
make test       # rebuild the warehouse on data/sample, then run pytest
```

No download is needed to try the warehouse: `make test` builds it on the small
sample committed in `data/sample`, which is what CI does.

`make help` lists every target; the later pipeline stages are added week by week.

## Design decisions

The decisions that shape everything else, with the numbers behind them
([EDA notebook](notebooks/01_eda.ipynb)):

1. [Target definition](docs/decisions/0001-target-definition.md): late means delivered after the promised calendar date; 6.77% of delivered orders.
2. [Prediction time](docs/decisions/0002-prediction-time.md): features must be known when the order is approved.
3. [Censoring cutoff](docs/decisions/0003-censoring-cutoff.md): the 10,201 labeled orders purchased after 2018-07-15 are dropped, because their slow deliveries are missing from the data.
4. [Chronological split](docs/decisions/0004-chronological-split.md): train on 2017, validate on Q1 2018, test on April to mid-July 2018. Validation runs at three times the test late rate, so models are never selected on it alone, and every fit only learns from outcomes already known when it would be used.
5. [Leakage rules](docs/decisions/0005-leakage-rules.md): nothing recorded after approval, and seller history only from orders already delivered.
6. [Calibration under shift](docs/decisions/0006-calibration-under-shift.md): sigmoid calibration fitted on the pooled folds and validation, chosen by replaying each option one period back in time.

## License

Code: [MIT](LICENSE). Data: CC BY-NC-SA 4.0, owned by Olist; see
[data/README.md](data/README.md).
