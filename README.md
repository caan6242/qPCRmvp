# qPCR Studio

A private Streamlit app for qPCR Ct/Cq analysis.

The app can now work with cleaned long-form data, many common raw qPCR export column names, and simple wide Ct tables. It calculates replicate QC, geometric-mean housekeeping normalisation on the expression scale, ΔCt, ΔΔCt, fold change, experiment-level summaries, trend interpretation, charts, and a downloadable Excel report.

You can upload up to 10 experiment files at once. If a file does not include an `Experiment`, `Run`, or `Plate` column, the app uses the filename as the experiment name so the batch can be analysed and compared automatically.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy online

Upload these files to your GitHub repo and redeploy on Streamlit Community Cloud.

## Recommended input format

| Experiment | Sample | Gene | Ct | Replicate |
|---|---|---|---:|---:|
| Exp 1 | Control | INS | 24.1 | 1 |
| Exp 1 | Control | INS | 24.3 | 2 |
| Exp 1 | HG | INS | 25.7 | 1 |
| Exp 2 | HG+PA | RPLP0 | 19.4 | 3 |

Minimum required biological fields:

- `Sample`
- `Gene` or `Target`
- `Ct`, `Cq`, or `Cp`

Optional but useful fields:

- `Experiment`, `Run`, `Plate`, or `Batch` for comparing repeated experiments
- `Replicate` or `Well` for traceability

Column names are case-insensitive. Extra columns are ignored.

## Raw-data recognition

The app recognises common qPCR export headings such as:

- sample: `Sample Name`, `Sample ID`, `Condition`, `Treatment`, `Group`
- target: `Target Name`, `Assay`, `Primer`, `Gene Symbol`
- Ct/Cq: `Ct`, `Cq`, `Cp`, `Cycle Threshold`, `Ct Mean`
- experiment: `Experiment`, `Run`, `Plate`, `Batch`
- replicate: `Replicate`, `Well`, `Position`

It also accepts a simple wide table where rows are samples and numeric gene columns contain Ct values. For Excel files with multiple sheets, each sheet is treated as a separate experiment if no experiment column is present.

## Calculations

Ct values are logarithmic. For multiple housekeeping genes, qPCR Studio averages housekeeping Ct values per sample and experiment, which is equivalent to using the geometric mean on the original expression scale.

ΔCt = target mean Ct - housekeeping mean Ct

ΔΔCt = sample ΔCt - control ΔCt for the same gene in the same experiment

Fold change = 2^-ΔΔCt

log2 fold change = -ΔΔCt

## QC behavior

The app does not remove an entire sample/gene group just because the full replicate spread is too large.

- If all replicates are within the cutoff, all are kept.
- If one replicate is an outlier, the app keeps the largest subset where max Ct - min Ct is within the cutoff.
- With triplicates, this usually means keeping the two closest replicates and removing the odd one out.
- A group only fails if it cannot find at least two replicates within the cutoff.
- Single replicates are allowed but flagged as `WARN_single_replicate`.

## Charts and insights

The app includes:

- fold-change bar plots
- log2 fold-change dot plots
- fold-change heatmap
- ΔCt plots
- raw Ct/Cq replicate box plots
- replicate QC spread plots
- experiment-to-experiment trend plots
- uploaded batch comparison across up to 10 experiments
- batch consistency ranking for reproducible vs variable trends
- ranked trend interpretation with QC-aware suggestions for follow-up experiments

The trend summaries are hypothesis-generating. Use them together with your assay design, biological replicates, controls, and domain knowledge.

## Past experiments

Use the `Export` tab to save the current analysis into the in-app experiment library. The `Past experiments` tab then lets you:

- review saved analyses with metadata
- compare selected experiments
- see cross-experiment trend summaries
- generate comparison charts
- export selected experiments together as one Excel workbook
- download the full experiment library as JSON
- import a previous library JSON file later

Streamlit Community Cloud does not provide reliable permanent local storage for app files, so the in-app library is remembered during the current browser/app session. Download the JSON library when you want to keep the history and import it next time.
