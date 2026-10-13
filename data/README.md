# Data

## Source

**Brazilian E-Commerce Public Dataset by Olist**, published by Olist on Kaggle:
<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce> (version 2).

About 100k orders placed on the Olist marketplace between 2016 and 2018, split
across nine CSV files (orders, order items, customers, sellers, products,
payments, reviews, geolocation, and a category-name translation table).

## License and attribution

The dataset is licensed **CC BY-NC-SA 4.0** (Attribution, NonCommercial,
ShareAlike). It belongs to Olist; this project only uses it for non-commercial,
educational purposes.

- `data/raw/` holds the full CSVs and is **gitignored**. They are never committed.
- `data/sample/` holds a small sample used by CI and tests. That sample is a
  derivative of the Olist dataset and stays under CC BY-NC-SA 4.0, not under
  this repository's MIT license, which covers the code only.

## The sample

`make sample` rebuilds it from `data/raw/` with
`src/opspilot/data/make_sample.py`:

- 2,020 orders: 95 from each month, chosen by a hash of the order id so the
  result is identical on every run, and all orders of the four near-empty
  months.
- Every related row: the items and payments of those orders, their customers,
  and the sellers and products those items refer to.
- Up to 3 geolocation points for each customer and seller zip prefix (the raw
  file has about 29 per prefix), plus the full category translation table.
- The reviews file is left out on purpose. Reviews are written after delivery
  and are never used.

The files keep the raw names and columns, so the same dbt sources read either
folder.

## Download

```bash
make data
```

This runs `src/opspilot/data/download.py`, which fetches the dataset with
`kagglehub`, copies the CSVs into `data/raw/`, prints each file's SHA-256 and
compares it with `data/raw.sha256`. The dataset is public, so no Kaggle account
or API token is required.
