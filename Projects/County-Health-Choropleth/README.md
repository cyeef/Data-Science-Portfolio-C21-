# County Health — Bivariate Choropleth (Poverty × Low Birthweight)

An interactive Streamlit map of U.S. counties showing **child poverty** and
**low birthweight** together, on a bivariate 3×3 colour grid. Counties high in
*both* measures light up in the darkest corner — the "pockets" where the two
burdens concentrate.

Built from the cleaned County Health Rankings (CHR) 2025 dataset published at
[`RaayGunz/county-health-maternal-clean`](https://huggingface.co/datasets/RaayGunz/county-health-maternal-clean).

## The finding this map illustrates

A regression + GaussianNB analysis of the CHR 2025 data (see the accompanying
notebook) found:

- **Child poverty is the dominant predictor** of low birthweight — regression
  coefficient **1.45**, roughly **4× any other feature**.
- **Air pollution (PM2.5) had a negligible effect** — correlation ~0.05,
  coefficient ~−0.10.
- Both a linear regression (R² 0.48 vs. dummy ~0.00) and a GaussianNB
  classifier (accuracy 0.76 vs. dummy 0.50) beat their baselines, confirming
  real signal.

This map is the visual companion to that result: it shows *where* poverty and
low birthweight overlap geographically.

## Features

- **Bivariate 3×3 map** — every county shaded by its poverty tier × low-birthweight tier.
- **Threshold-highlight mode** — set your own poverty and low-birthweight cut-offs with sliders; the map highlights counties meeting *both*, with a live count.
- **State zoom** — focus on any single state.
- **Hover tooltips** — county name, both values, and tier labels.

## Run locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The app pulls its data at runtime from the Hugging Face dataset and the county
boundaries from Plotly's public GeoJSON, so no local data file is needed.

## Data & license

Source: County Health Rankings & Roadmaps 2025 (University of Wisconsin
Population Health Institute). The cleaned dataset is shared for **educational
and non-commercial** use under **CC-BY-NC-4.0**, consistent with the
[CHR&R Terms of Use](https://www.countyhealthrankings.org/terms-use).

County-level associations only — not causal or individual-level claims.
