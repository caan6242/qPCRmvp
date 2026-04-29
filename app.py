import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DISPLAY_COLUMNS = ["Experiment", "Sample", "Gene", "Ct"]
GROUP_COLUMNS = ["Experiment", "Sample", "Gene"]

ALIASES = {
    "sample": [
        "sample",
        "sample name",
        "sample_name",
        "sample id",
        "sampleid",
        "specimen",
        "condition",
        "treatment",
        "group",
    ],
    "gene": [
        "gene",
        "target",
        "target name",
        "target_name",
        "assay",
        "assay name",
        "primer",
        "primer name",
        "detector",
        "gene symbol",
    ],
    "ct": [
        "ct",
        "cq",
        "cp",
        "cycle threshold",
        "threshold cycle",
        "ct mean",
        "cq mean",
        "ct value",
        "cq value",
    ],
    "experiment": [
        "experiment",
        "experiment id",
        "experiment name",
        "run",
        "run id",
        "plate",
        "plate id",
        "batch",
        "study",
    ],
    "replicate": ["replicate", "rep", "well", "well position", "position"],
}


@dataclass
class AnalysisSettings:
    control_sample: str
    housekeeping_genes: List[str]
    outlier_cutoff: float
    apply_qc: bool


def column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_column(df: pd.DataFrame, role: str) -> Optional[str]:
    keyed = {column_key(c): c for c in df.columns}
    for alias in ALIASES[role]:
        if column_key(alias) in keyed:
            return keyed[column_key(alias)]

    # Gentle fallback for common exports with verbose headings.
    candidates = []
    for col in df.columns:
        key = column_key(col)
        if role == "sample" and "sample" in key:
            candidates.append(col)
        elif role == "gene" and any(token in key for token in ["target", "gene", "assay"]):
            candidates.append(col)
        elif role == "ct" and key in {"ct", "cq", "cp"}:
            candidates.append(col)
        elif role == "experiment" and any(token in key for token in ["experiment", "plate", "run", "batch"]):
            candidates.append(col)
        elif role == "replicate" and any(token in key for token in ["replicate", "well", "position"]):
            candidates.append(col)
    return candidates[0] if candidates else None


def infer_wide_ct_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    sample_col = find_column(df, "sample")
    if sample_col is None:
        return None

    experiment_col = find_column(df, "experiment")
    id_cols = [sample_col]
    if experiment_col and experiment_col not in id_cols:
        id_cols.append(experiment_col)

    numeric_cols = []
    for col in df.columns:
        if col in id_cols:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() >= max(2, int(len(df) * 0.5)):
            numeric_cols.append(col)

    if len(numeric_cols) < 2:
        return None

    melted = df[id_cols + numeric_cols].melt(
        id_vars=id_cols,
        value_vars=numeric_cols,
        var_name="Gene",
        value_name="Ct",
    )
    melted = melted.rename(columns={sample_col: "Sample"})
    if experiment_col:
        melted = melted.rename(columns={experiment_col: "Experiment"})
    return melted


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    sample_col = find_column(df, "sample")
    gene_col = find_column(df, "gene")
    ct_col = find_column(df, "ct")
    experiment_col = find_column(df, "experiment")
    replicate_col = find_column(df, "replicate")

    if not all([sample_col, gene_col, ct_col]):
        wide = infer_wide_ct_table(df)
        if wide is None:
            missing = []
            if sample_col is None:
                missing.append("Sample")
            if gene_col is None:
                missing.append("Gene/Target")
            if ct_col is None:
                missing.append("Ct/Cq")
            raise ValueError(
                "Could not recognise the input format. I need long-form qPCR data with "
                "sample, gene/target and Ct/Cq columns, or a wide table with samples in rows "
                f"and genes in numeric Ct columns. Missing: {', '.join(missing)}."
            )
        return normalise_columns(wide)

    keep = [sample_col, gene_col, ct_col]
    rename = {sample_col: "Sample", gene_col: "Gene", ct_col: "Ct"}
    if experiment_col and experiment_col not in keep:
        keep.append(experiment_col)
        rename[experiment_col] = "Experiment"
    if replicate_col and replicate_col not in keep:
        keep.append(replicate_col)
        rename[replicate_col] = "Replicate"

    out = df[keep].rename(columns=rename).copy()
    if "Experiment" not in out.columns:
        out["Experiment"] = "Experiment 1"
    if "Replicate" not in out.columns:
        out["Replicate"] = ""

    for col in ["Experiment", "Sample", "Gene", "Replicate"]:
        out[col] = out[col].astype(str).str.strip()
    out["Ct"] = pd.to_numeric(out["Ct"], errors="coerce")
    out = out.dropna(subset=["Sample", "Gene", "Ct"])
    out = out[(out["Sample"] != "") & (out["Gene"] != "")]
    out.loc[out["Experiment"].isin(["", "nan", "None"]), "Experiment"] = "Experiment 1"
    return out[["Experiment", "Sample", "Gene", "Ct", "Replicate"]].reset_index(drop=True)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        usable = []
        for sheet_name, sheet in sheets.items():
            if sheet.dropna(how="all").empty:
                continue
            sheet = sheet.dropna(how="all").copy()
            if find_column(sheet, "experiment") is None:
                sheet["Experiment"] = sheet_name
            usable.append(sheet)
        if not usable:
            raise ValueError("The Excel file did not contain any usable sheets.")
        return pd.concat(usable, ignore_index=True)
    raise ValueError("Please upload a CSV or Excel file.")


def make_example_data() -> pd.DataFrame:
    rows = []
    experiments = ["Exp 1", "Exp 2"]
    samples = ["Control", "HG", "PA", "HG+PA"]
    genes = ["RPLP0", "TBP", "INS", "IAPP", "PCSK1", "HSPA5"]
    base_ct = {
        "RPLP0": 18.5,
        "TBP": 22.0,
        "INS": 24.0,
        "IAPP": 25.2,
        "PCSK1": 27.0,
        "HSPA5": 26.5,
    }
    effects = {
        "Control": {},
        "HG": {"INS": 0.6, "IAPP": 0.4, "PCSK1": -0.2, "HSPA5": -0.8},
        "PA": {"INS": 0.9, "IAPP": 0.5, "PCSK1": 0.1, "HSPA5": -1.0},
        "HG+PA": {"INS": 1.3, "IAPP": 0.9, "PCSK1": -1.2, "HSPA5": -1.8},
    }
    rng = np.random.default_rng(7)
    for exp_idx, experiment in enumerate(experiments):
        exp_shift = rng.normal(0, 0.12)
        for sample in samples:
            for gene in genes:
                for rep in range(1, 4):
                    ct = (
                        base_ct[gene]
                        + effects.get(sample, {}).get(gene, 0)
                        + exp_shift
                        + rng.normal(0, 0.12)
                    )
                    rows.append(
                        {
                            "Experiment": experiment,
                            "Sample": sample,
                            "Gene": gene,
                            "Ct": round(float(ct), 3),
                            "Replicate": rep,
                        }
                    )

    rows.append({"Experiment": "Exp 1", "Sample": "Control", "Gene": "TBP", "Ct": 25.7, "Replicate": 4})
    return pd.DataFrame(rows)


def select_replicates_for_group(values: pd.Series, cutoff: float) -> pd.Series:
    """
    Keep the largest subset of Ct values where max-min <= cutoff.
    If several subsets have the same size, keep the subset with smallest spread.
    """
    vals = values.dropna().sort_values()

    if len(vals) <= 1:
        return pd.Series(True, index=values.index)

    best_index = None
    best_n = 0
    best_spread = np.inf

    sorted_items = list(vals.items())
    for i in range(len(sorted_items)):
        for j in range(i, len(sorted_items)):
            subset = sorted_items[i : j + 1]
            subset_values = [v for _, v in subset]
            spread = max(subset_values) - min(subset_values)
            n = len(subset)
            if spread <= cutoff and n >= 2:
                if n > best_n or (n == best_n and spread < best_spread):
                    best_index = [idx for idx, _ in subset]
                    best_n = n
                    best_spread = spread

    keep = pd.Series(False, index=values.index)
    if best_index is not None:
        keep.loc[best_index] = True
    return keep


def replicate_qc(raw: pd.DataFrame, settings: AnalysisSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    keep_mask = pd.Series(False, index=raw.index)

    for keys, group in raw.groupby(GROUP_COLUMNS):
        experiment, sample, gene = keys
        values = group["Ct"]
        original_n = len(values)
        original_min = values.min()
        original_max = values.max()
        original_spread = original_max - original_min if original_n else np.nan

        if settings.apply_qc:
            group_keep = select_replicates_for_group(values, settings.outlier_cutoff)
        else:
            group_keep = pd.Series(True, index=values.index)

        kept_values = values[group_keep]
        excluded_values = values[~group_keep]

        n_used = len(kept_values)
        if original_n == 1:
            qc_status = "WARN_single_replicate"
            use_group = True
        elif n_used >= 2:
            qc_status = "PASS" if len(excluded_values) == 0 else "PASS_outlier_removed"
            use_group = True
        else:
            qc_status = "FAIL_no_two_replicates_within_cutoff"
            use_group = False

        if use_group:
            keep_mask.loc[kept_values.index] = True

        rows.append(
            {
                "Experiment": experiment,
                "Sample": sample,
                "Gene": gene,
                "n_replicates_original": original_n,
                "n_used": n_used if use_group else 0,
                "n_excluded": len(excluded_values) if use_group else original_n,
                "min_ct_original": original_min,
                "max_ct_original": original_max,
                "ct_spread_original": original_spread,
                "mean_ct_used": kept_values.mean() if use_group else np.nan,
                "sd_ct_used": kept_values.std() if use_group else np.nan,
                "ct_values_used": ", ".join(f"{x:.3f}" for x in kept_values.tolist()) if use_group else "",
                "ct_values_excluded": ", ".join(f"{x:.3f}" for x in excluded_values.tolist()) if len(excluded_values) else "",
                "qc_status": qc_status,
            }
        )

    qc = pd.DataFrame(rows)
    clean = raw.loc[keep_mask].copy() if settings.apply_qc else raw.copy()
    return qc, clean


def calculate_results(clean: pd.DataFrame, settings: AnalysisSettings) -> Dict[str, pd.DataFrame]:
    mean_ct = clean.groupby(GROUP_COLUMNS, as_index=False).agg(
        mean_ct=("Ct", "mean"),
        sd_ct=("Ct", "std"),
        n_used=("Ct", "count"),
    )

    hk = mean_ct[mean_ct["Gene"].isin(settings.housekeeping_genes)].copy()
    if hk.empty:
        raise ValueError("None of the selected housekeeping genes were found in the data.")

    hk_summary = hk.groupby(["Experiment", "Sample"], as_index=False).agg(
        housekeeping_mean_ct=("mean_ct", "mean"),
        housekeeping_genes_used=("Gene", lambda x: ", ".join(sorted(x.unique()))),
        n_housekeeping_genes=("Gene", "nunique"),
    )

    targets = mean_ct[~mean_ct["Gene"].isin(settings.housekeeping_genes)].copy()
    if targets.empty:
        raise ValueError("No target genes left after removing housekeeping genes.")

    delta_ct = targets.merge(hk_summary, on=["Experiment", "Sample"], how="left")
    delta_ct["delta_ct"] = delta_ct["mean_ct"] - delta_ct["housekeeping_mean_ct"]

    missing_hk = delta_ct[delta_ct["housekeeping_mean_ct"].isna()][["Experiment", "Sample"]].drop_duplicates()
    if not missing_hk.empty:
        pairs = [f"{row.Experiment}/{row.Sample}" for row in missing_hk.itertuples()]
        raise ValueError(
            "Some experiment/sample groups are missing usable housekeeping data after QC: "
            + ", ".join(pairs)
            + ". Check the QC tab for failed housekeeping groups."
        )

    control = delta_ct[delta_ct["Sample"] == settings.control_sample]
    if control.empty:
        raise ValueError(f"Control sample '{settings.control_sample}' was not found among target-gene rows.")

    control_ref = control.groupby(["Experiment", "Gene"], as_index=False).agg(
        control_delta_ct=("delta_ct", "mean")
    )

    ddct = delta_ct.merge(control_ref, on=["Experiment", "Gene"], how="left")
    missing_control = ddct[ddct["control_delta_ct"].isna()]["Experiment"].unique()
    if len(missing_control) > 0:
        raise ValueError(
            f"Control sample '{settings.control_sample}' is missing for these experiment(s): "
            + ", ".join(map(str, missing_control))
            + ". Add a control to each experiment or remove the experiment column."
        )

    ddct["delta_delta_ct"] = ddct["delta_ct"] - ddct["control_delta_ct"]
    ddct["fold_change"] = np.power(2, -ddct["delta_delta_ct"])
    ddct["log2_fold_change"] = -ddct["delta_delta_ct"]

    cols = [
        "Experiment",
        "Sample",
        "Gene",
        "n_used",
        "mean_ct",
        "sd_ct",
        "housekeeping_mean_ct",
        "housekeeping_genes_used",
        "delta_ct",
        "control_delta_ct",
        "delta_delta_ct",
        "log2_fold_change",
        "fold_change",
    ]
    ddct = ddct[cols].sort_values(["Gene", "Sample", "Experiment"]).reset_index(drop=True)

    across_experiments = ddct.groupby(["Sample", "Gene"], as_index=False).agg(
        experiments=("Experiment", "nunique"),
        mean_fold_change=("fold_change", "mean"),
        median_fold_change=("fold_change", "median"),
        sd_fold_change=("fold_change", "std"),
        mean_log2_fold_change=("log2_fold_change", "mean"),
        sd_log2_fold_change=("log2_fold_change", "std"),
        min_fold_change=("fold_change", "min"),
        max_fold_change=("fold_change", "max"),
    )
    across_experiments["direction"] = np.select(
        [
            across_experiments["mean_log2_fold_change"] >= 0.58,
            across_experiments["mean_log2_fold_change"] <= -0.58,
        ],
        ["up", "down"],
        default="stable/mild",
    )

    return {
        "mean_ct": mean_ct.sort_values(["Experiment", "Sample", "Gene"]).reset_index(drop=True),
        "housekeeping_summary": hk_summary.sort_values(["Experiment", "Sample"]).reset_index(drop=True),
        "delta_ct": delta_ct.sort_values(["Gene", "Sample", "Experiment"]).reset_index(drop=True),
        "final_results": ddct,
        "across_experiments": across_experiments.sort_values(
            ["Gene", "Sample"]
        ).reset_index(drop=True),
    }


def classify_trend(log2_fc: float, experiments: int, sd_log2: float) -> Tuple[str, str]:
    abs_log2 = abs(log2_fc)
    if abs_log2 >= 1:
        strength = "strong"
    elif abs_log2 >= 0.58:
        strength = "moderate"
    elif abs_log2 >= 0.32:
        strength = "mild"
    else:
        strength = "little change"

    direction = "higher expression" if log2_fc > 0 else "lower expression" if log2_fc < 0 else "no change"
    if experiments <= 1:
        confidence = "single experiment"
    elif pd.isna(sd_log2) or sd_log2 <= 0.5:
        confidence = "consistent"
    elif sd_log2 <= 1:
        confidence = "variable"
    else:
        confidence = "inconsistent"
    return f"{strength} {direction}", confidence


def build_insights(final: pd.DataFrame, qc: pd.DataFrame, settings: AnalysisSettings) -> Tuple[pd.DataFrame, List[str]]:
    non_control = final[final["Sample"] != settings.control_sample].copy()
    if non_control.empty:
        return pd.DataFrame(), ["Add at least one non-control sample to generate biological trend summaries."]

    summary = non_control.groupby(["Sample", "Gene"], as_index=False).agg(
        experiments=("Experiment", "nunique"),
        mean_fold_change=("fold_change", "mean"),
        median_fold_change=("fold_change", "median"),
        mean_log2_fold_change=("log2_fold_change", "mean"),
        sd_log2_fold_change=("log2_fold_change", "std"),
        min_fold_change=("fold_change", "min"),
        max_fold_change=("fold_change", "max"),
    )
    classified = summary.apply(
        lambda row: classify_trend(row["mean_log2_fold_change"], row["experiments"], row["sd_log2_fold_change"]),
        axis=1,
        result_type="expand",
    )
    summary["trend"] = classified[0]
    summary["confidence"] = classified[1]
    summary["interpretation"] = summary.apply(
        lambda row: (
            f"{row.Sample} shows {row.trend} for {row.Gene} "
            f"(mean FC {row.mean_fold_change:.2f}, {row.experiments} experiment(s), {row.confidence})."
        ),
        axis=1,
    )
    summary["suggested_next_step"] = summary.apply(make_suggestion, axis=1)
    summary = summary.sort_values("mean_log2_fold_change", key=lambda s: s.abs(), ascending=False)

    notes = []
    high_spread = qc[qc["ct_spread_original"] > settings.outlier_cutoff]
    if not high_spread.empty:
        notes.append(
            f"{len(high_spread)} sample/gene group(s) exceeded the replicate spread cutoff; inspect QC before over-interpreting those genes."
        )
    single_reps = qc[qc["qc_status"] == "WARN_single_replicate"]
    if not single_reps.empty:
        notes.append(
            f"{len(single_reps)} sample/gene group(s) have a single replicate; repeat technical replicates for stronger evidence."
        )
    if summary["confidence"].isin(["variable", "inconsistent", "single experiment"]).any():
        notes.append(
            "Prioritise repeating the largest or most variable effects before treating them as stable biology."
        )
    notes.append(
        "These summaries describe expression patterns from the uploaded data; they are hypothesis-generating and should be interpreted with your assay design and biology."
    )
    return summary.reset_index(drop=True), notes


def make_suggestion(row: pd.Series) -> str:
    gene = row["Gene"]
    sample = row["Sample"]
    log2_fc = row["mean_log2_fold_change"]
    confidence = row["confidence"]

    if confidence in {"single experiment", "variable", "inconsistent"}:
        return f"Repeat {sample} for {gene} with biological replicates and confirm housekeeping stability."
    if log2_fc >= 1:
        return f"Validate {gene} induction in {sample}; consider a dose/time-course and a pathway marker panel."
    if log2_fc <= -1:
        return f"Validate {gene} suppression in {sample}; test whether the effect is rescued or strengthened under related conditions."
    if abs(log2_fc) >= 0.58:
        return f"Follow up {gene} in {sample} with more biological replicates to see if the moderate effect is reproducible."
    return f"Keep {gene} in view, but prioritise stronger or more consistent effects for immediate follow-up."


def make_excel_report(raw, qc, clean, outputs, insights) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        raw.to_excel(writer, index=False, sheet_name="Recognised_Raw_Data")
        qc.to_excel(writer, index=False, sheet_name="QC_Summary")
        clean.to_excel(writer, index=False, sheet_name="Clean_Data_Used")
        outputs["mean_ct"].to_excel(writer, index=False, sheet_name="Mean_Ct")
        outputs["housekeeping_summary"].to_excel(writer, index=False, sheet_name="Housekeeping")
        outputs["delta_ct"].to_excel(writer, index=False, sheet_name="Delta_Ct")
        outputs["final_results"].to_excel(writer, index=False, sheet_name="Fold_Change")
        outputs["across_experiments"].to_excel(writer, index=False, sheet_name="Experiment_Summary")
        if insights is not None and not insights.empty:
            insights.to_excel(writer, index=False, sheet_name="Insights")

        workbook = writer.book
        number_fmt = workbook.add_format({"num_format": "0.000"})
        header_fmt = workbook.add_format({"bold": True})
        for sheet in writer.sheets.values():
            sheet.set_row(0, None, header_fmt)
            sheet.set_column(0, 30, 18, number_fmt)

    return buffer.getvalue()


def add_heatmap(plot_df: pd.DataFrame) -> None:
    heat = plot_df.groupby(["Sample", "Gene"], as_index=False).agg(log2_fold_change=("log2_fold_change", "mean"))
    fig = px.imshow(
        heat.pivot(index="Gene", columns="Sample", values="log2_fold_change"),
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        aspect="auto",
        title="Mean log2 fold change heatmap",
        labels={"color": "log2 FC"},
    )
    st.plotly_chart(fig, use_container_width=True)


def app():
    st.set_page_config(page_title="qPCR Studio", layout="wide")

    st.title("qPCR Studio")
    st.caption(
        "Upload cleaned or raw qPCR Ct/Cq data -> QC -> ΔCt -> ΔΔCt -> fold change -> trends -> charts -> Excel report."
    )

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        use_example = st.toggle("Use example data", value=uploaded is None)

        st.header("Analysis settings")
        st.write("Upload data first to select samples and genes.")

    try:
        if uploaded is not None:
            raw_input = read_uploaded_file(uploaded)
        elif use_example:
            raw_input = make_example_data()
        else:
            st.info("Upload a CSV/Excel file or enable example data.")
            return

        raw = normalise_columns(raw_input)

        samples = sorted(raw["Sample"].unique().tolist())
        genes = sorted(raw["Gene"].unique().tolist())
        experiments = sorted(raw["Experiment"].unique().tolist())

        with st.sidebar:
            default_control_idx = samples.index("Control") if "Control" in samples else 0
            control_sample = st.selectbox("Control/reference sample", samples, index=default_control_idx)

            default_hk = [g for g in ["RPLP0", "TBP", "PPIA", "HPRT", "HPRT1", "GAPDH", "ACTB", "PBGD"] if g in genes]
            housekeeping_genes = st.multiselect(
                "Housekeeping genes",
                genes,
                default=default_hk if default_hk else genes[:1],
            )

            outlier_cutoff = st.number_input(
                "Ct cutoff for keeping replicate values together",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="The app keeps the largest set of replicates where max Ct - min Ct is within this cutoff.",
            )
            apply_qc = st.checkbox(
                "Apply QC and remove outlier replicates",
                value=True,
                help="With triplicates, one outlier can be removed while the remaining two are still used.",
            )

        settings = AnalysisSettings(
            control_sample=control_sample,
            housekeeping_genes=housekeeping_genes,
            outlier_cutoff=float(outlier_cutoff),
            apply_qc=apply_qc,
        )

        if not settings.housekeeping_genes:
            st.warning("Select at least one housekeeping gene.")
            return

        qc, clean = replicate_qc(raw, settings)
        outputs = calculate_results(clean, settings)
        final = outputs["final_results"]
        insights, insight_notes = build_insights(final, qc, settings)

        top1, top2, top3, top4, top5 = st.columns(5)
        top1.metric("Rows uploaded", len(raw))
        top2.metric("Rows used", len(clean))
        top3.metric("Experiments", len(experiments))
        top4.metric("Sample/gene groups", len(qc))
        top5.metric("Outlier values removed", int(qc["n_excluded"].sum()) if apply_qc else 0)

        with st.expander("Recognised input structure", expanded=False):
            st.write(
                f"Detected {len(experiments)} experiment(s), {len(samples)} sample(s), "
                f"{len(genes)} gene/target(s)."
            )
            st.dataframe(raw.head(50), use_container_width=True)

        tabs = st.tabs(["Results", "Insights", "Plots", "QC", "Raw/Clean data", "Export"])

        with tabs[0]:
            st.subheader("Fold-change results")
            st.dataframe(final, use_container_width=True)

            st.subheader("Across-experiment summary")
            st.dataframe(outputs["across_experiments"], use_container_width=True)

            st.subheader("Housekeeping summary")
            st.dataframe(outputs["housekeeping_summary"], use_container_width=True)

        with tabs[1]:
            st.subheader("Trend interpretation")
            if insights.empty:
                st.info("No non-control trends to summarise yet.")
            else:
                st.dataframe(insights, use_container_width=True)
                strongest = insights.iloc[0]
                st.info(strongest["interpretation"] + " " + strongest["suggested_next_step"])

            st.subheader("QC-aware notes")
            for note in insight_notes:
                st.write(f"- {note}")

        with tabs[2]:
            st.subheader("Charts")
            selected_genes = st.multiselect(
                "Genes to plot",
                sorted(final["Gene"].unique()),
                default=sorted(final["Gene"].unique())[: min(6, final["Gene"].nunique())],
            )
            selected_samples = st.multiselect(
                "Samples to plot",
                sorted(final["Sample"].unique()),
                default=sorted(final["Sample"].unique()),
            )
            plot_df = final[final["Gene"].isin(selected_genes) & final["Sample"].isin(selected_samples)].copy()

            if not plot_df.empty:
                chart_tabs = st.tabs(
                    [
                        "Fold change",
                        "Heatmap",
                        "ΔCt",
                        "Raw Ct",
                        "Replicate QC",
                        "Experiments",
                    ]
                )
                with chart_tabs[0]:
                    fig = px.bar(
                        plot_df,
                        x="Sample",
                        y="fold_change",
                        color="Gene",
                        facet_col="Experiment" if plot_df["Experiment"].nunique() > 1 else None,
                        barmode="group",
                        hover_data=["delta_ct", "delta_delta_ct", "mean_ct", "log2_fold_change"],
                        title="Fold change by sample and gene",
                    )
                    fig.add_hline(y=1, line_dash="dash")
                    st.plotly_chart(fig, use_container_width=True)

                    fig_log = px.scatter(
                        plot_df,
                        x="Sample",
                        y="log2_fold_change",
                        color="Gene",
                        symbol="Experiment",
                        hover_data=["fold_change", "delta_delta_ct"],
                        title="log2 fold change dot plot",
                    )
                    fig_log.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig_log, use_container_width=True)

                with chart_tabs[1]:
                    add_heatmap(plot_df)

                with chart_tabs[2]:
                    fig_delta = px.bar(
                        plot_df,
                        x="Sample",
                        y="delta_ct",
                        color="Gene",
                        facet_col="Experiment" if plot_df["Experiment"].nunique() > 1 else None,
                        barmode="group",
                        hover_data=["mean_ct", "housekeeping_mean_ct"],
                        title="ΔCt by sample and gene",
                    )
                    st.plotly_chart(fig_delta, use_container_width=True)

                with chart_tabs[3]:
                    raw_plot = clean[clean["Gene"].isin(selected_genes) & clean["Sample"].isin(selected_samples)]
                    fig_ct = px.box(
                        raw_plot,
                        x="Sample",
                        y="Ct",
                        color="Gene",
                        points="all",
                        facet_col="Experiment" if raw_plot["Experiment"].nunique() > 1 else None,
                        title="Raw Ct/Cq replicate distribution",
                    )
                    fig_ct.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_ct, use_container_width=True)

                with chart_tabs[4]:
                    qc_plot = qc[qc["Gene"].isin(selected_genes) & qc["Sample"].isin(selected_samples)].copy()
                    fig_qc = px.bar(
                        qc_plot,
                        x="Sample",
                        y="ct_spread_original",
                        color="qc_status",
                        facet_col="Gene",
                        hover_data=["Experiment", "n_replicates_original", "n_used", "ct_values_excluded"],
                        title="Replicate Ct spread before QC",
                    )
                    fig_qc.add_hline(y=settings.outlier_cutoff, line_dash="dash")
                    st.plotly_chart(fig_qc, use_container_width=True)

                with chart_tabs[5]:
                    if plot_df["Experiment"].nunique() > 1:
                        fig_exp = px.line(
                            plot_df.sort_values("Experiment"),
                            x="Experiment",
                            y="fold_change",
                            color="Sample",
                            line_group="Sample",
                            facet_col="Gene",
                            markers=True,
                            hover_data=["log2_fold_change", "delta_delta_ct"],
                            title="Fold-change trend across experiments",
                        )
                        fig_exp.add_hline(y=1, line_dash="dash")
                        st.plotly_chart(fig_exp, use_container_width=True)
                    else:
                        st.info("Add an Experiment/Run/Plate column or upload a multi-sheet Excel file to compare experiments.")
            else:
                st.info("Select at least one gene and sample to plot.")

        with tabs[3]:
            st.subheader("Replicate QC")
            st.dataframe(qc, use_container_width=True)

            flagged = qc[qc["qc_status"] != "PASS"]
            if not flagged.empty:
                st.warning("Some groups had outliers, single replicates, or failed QC.")
                st.dataframe(flagged, use_container_width=True)
            else:
                st.success("All replicate groups passed without outlier removal.")

        with tabs[4]:
            left, right = st.columns(2)
            with left:
                st.subheader("Recognised raw data")
                st.dataframe(raw, use_container_width=True)
            with right:
                st.subheader("Clean data used")
                st.dataframe(clean, use_container_width=True)

        with tabs[5]:
            st.subheader("Download report")
            excel_bytes = make_excel_report(raw, qc, clean, outputs, insights)
            st.download_button(
                label="Download Excel report",
                data=excel_bytes,
                file_name="qpcr_studio_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.subheader("Copy final results as CSV")
            st.code(final.to_csv(index=False), language="csv")

    except Exception as e:
        st.error(str(e))


if __name__ == "__main__":
    app()
