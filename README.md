# qPCR Studio MVP

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
- replicate spread
- optional exclusion when replicate spread exceeds cutoff
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

Then open the local URL Streamlit gives you.

## Deploy online

The easiest route is Streamlit Community Cloud:

1. Put these files in a GitHub repo.
2. Go to Streamlit Community Cloud.
3. Connect the repo.
4. Set the app file to `app.py`.
5. Deploy.

For private/lab use, deploy behind authentication or use a private hosting option.

## Input rules

Required columns:

- `Sample`
- `Gene`
- `Ct`

Column names are case-insensitive. Extra columns are ignored for calculations.

## Notes on housekeeping normalization

Ct values are logarithmic. For multiple housekeeping genes, this app uses the arithmetic mean of housekeeping Ct values per sample, which is equivalent to using the geometric mean on the original expression scale.

ΔCt = target mean Ct - housekeeping mean Ct

ΔΔCt = sample ΔCt - mean control ΔCt for the same gene

Fold change = 2^-ΔΔCt
