# qPCR Studio MVP v2

A private web app for cleaned qPCR Ct data.

Input format:

| Sample | Gene | Ct |
|---|---|---|
| Control | INS | 24.1 |
| Control | INS | 24.3 |
| HG | INS | 25.7 |
| HG+PA | RPLP0 | 19.4 |

The app calculates:

- replicate QC
- outlier replicate removal
- mean Ct
- housekeeping mean Ct
- ΔCt
- ΔΔCt against a selected control sample
- fold change using 2^-ΔΔCt
- plots per gene
- downloadable Excel report

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy online

Upload these files to your GitHub repo and redeploy on Streamlit Community Cloud.

## Required columns

- `Sample`
- `Gene`
- `Ct`

Column names are case-insensitive. Extra columns are ignored.

## v2 QC behavior

This version does **not** remove an entire sample/gene group just because the full replicate spread is too large.

Instead:

- If all replicates are within the cutoff, all are kept.
- If one replicate is an outlier, the app keeps the largest subset where max Ct - min Ct is within the cutoff.
- With triplicates, this usually means keeping the two closest replicates and removing the odd one out.
- A group only fails if it cannot find at least two replicates within the cutoff.
- Single replicates are allowed but flagged as `WARN_single_replicate`.

## Normalization

Ct values are logarithmic. For multiple housekeeping genes, this app uses the arithmetic mean of housekeeping Ct values per sample, which is equivalent to using the geometric mean on the original expression scale.

ΔCt = target mean Ct - housekeeping mean Ct

ΔΔCt = sample ΔCt - mean control ΔCt for the same gene

Fold change = 2^-ΔΔCt
