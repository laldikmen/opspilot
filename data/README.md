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
- `data/sample/` holds a small sample (about 2,000 orders and their related rows)
  used by CI and tests. That sample is a derivative of the Olist dataset and stays
  under CC BY-NC-SA 4.0, not under this repository's MIT license, which covers the
  code only.

## Download

```bash
make data
```

This runs `src/opspilot/data/download.py`, which fetches the dataset with
`kagglehub`, copies the CSVs into `data/raw/`, prints each file's SHA-256 and
compares it with `data/raw.sha256`. The dataset is public, so no Kaggle account
or API token is required.
