import io
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


REQUIRED_COLUMNS = ["Sample", "Gene", "Ct"]


@dataclass
class AnalysisSettings:
    control_sample: str
    housekeeping_genes: List[str]
    outlier_cutoff: float
    apply_qc: bool


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in REQUIRED_COLUMNS if c.lower() not in col_map]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    out = df[[col_map["sample"], col_map["gene"], col_map["ct"]]].copy()
    out.columns = REQUIRED_COLUMNS

    out["Sample"] = out["Sample"].astype(str).str.strip()
    out["Gene"] = out["Gene"].astype(str).str.strip()
    out["Ct"] = pd.to_numeric(out["Ct"], errors="coerce")
    out = out.dropna(subset=["Sample", "Gene", "Ct"])
    out = out[(out["Sample"] != "") & (out["Gene"] != "")]
    return out


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or Excel file.")


def make_example_data() -> pd.DataFrame:
    rows = []
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
    for sample in samples:
        for gene in genes:
            for _ in range(3):
                ct = base_ct[gene] + effects.get(sample, {}).get(gene, 0) + rng.normal(0, 0.12)
                rows.append({"Sample": sample, "Gene": gene, "Ct": round(float(ct), 3)})

    # Add one deliberate outlier to demonstrate v2 QC.
    rows.append({"Sample": "Control", "Gene": "TBP", "Ct": 25.7})
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
    """
    QC rule:
    - If all replicates fit within the cutoff, keep all.
    - If one or more values are outside, keep the largest subset with spread <= cutoff.
    - If at least two replicates remain, calculate using those.
    - If no pair fits within cutoff, exclude that sample/gene group from calculations.
    """
    rows = []
    keep_mask = pd.Series(False, index=raw.index)

    for (sample, gene), group in raw.groupby(["Sample", "Gene"]):
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
            if len(excluded_values) == 0:
                qc_status = "PASS"
            else:
                qc_status = "PASS_outlier_removed"
            use_group = True
        else:
            qc_status = "FAIL_no_two_replicates_within_cutoff"
            use_group = False

        if use_group:
            keep_mask.loc[kept_values.index] = True

        rows.append({
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
        })

    qc = pd.DataFrame(rows)
    clean = raw.loc[keep_mask].copy() if settings.apply_qc else raw.copy()
    return qc, clean


def calculate_results(clean: pd.DataFrame, settings: AnalysisSettings) -> Dict[str, pd.DataFrame]:
    mean_ct = clean.groupby(["Sample", "Gene"], as_index=False).agg(
        mean_ct=("Ct", "mean"),
        sd_ct=("Ct", "std"),
        n_used=("Ct", "count"),
    )

    hk = mean_ct[mean_ct["Gene"].isin(settings.housekeeping_genes)].copy()
    if hk.empty:
        raise ValueError("None of the selected housekeeping genes were found in the data.")

    hk_summary = hk.groupby("Sample", as_index=False).agg(
        housekeeping_mean_ct=("mean_ct", "mean"),
        housekeeping_genes_used=("Gene", lambda x: ", ".join(sorted(x.unique()))),
        n_housekeeping_genes=("Gene", "nunique"),
    )

    targets = mean_ct[~mean_ct["Gene"].isin(settings.housekeeping_genes)].copy()
    if targets.empty:
        raise ValueError("No target genes left after removing housekeeping genes.")

    delta_ct = targets.merge(hk_summary, on="Sample", how="left")
    delta_ct["delta_ct"] = delta_ct["mean_ct"] - delta_ct["housekeeping_mean_ct"]

    missing_hk = delta_ct[delta_ct["housekeeping_mean_ct"].isna()]["Sample"].unique()
    if len(missing_hk) > 0:
        raise ValueError(
            "Some samples are missing usable housekeeping data after QC: "
            + ", ".join(map(str, missing_hk))
            + ". Check the QC tab for failed housekeeping groups."
        )

    control = delta_ct[delta_ct["Sample"] == settings.control_sample]
    if control.empty:
        raise ValueError(f"Control sample '{settings.control_sample}' was not found among target-gene rows.")

    control_ref = control.groupby("Gene", as_index=False).agg(
        control_delta_ct=("delta_ct", "mean")
    )

    ddct = delta_ct.merge(control_ref, on="Gene", how="left")
    ddct["delta_delta_ct"] = ddct["delta_ct"] - ddct["control_delta_ct"]
    ddct["fold_change"] = np.power(2, -ddct["delta_delta_ct"])

    cols = [
        "Sample", "Gene", "n_used", "mean_ct", "sd_ct",
        "housekeeping_mean_ct", "housekeeping_genes_used",
        "delta_ct", "control_delta_ct", "delta_delta_ct", "fold_change"
    ]
    ddct = ddct[cols].sort_values(["Gene", "Sample"]).reset_index(drop=True)

    return {
        "mean_ct": mean_ct.sort_values(["Sample", "Gene"]).reset_index(drop=True),
        "housekeeping_summary": hk_summary.sort_values("Sample").reset_index(drop=True),
        "delta_ct": delta_ct.sort_values(["Gene", "Sample"]).reset_index(drop=True),
        "final_results": ddct,
    }


def make_excel_report(raw, qc, clean, outputs) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        raw.to_excel(writer, index=False, sheet_name="Raw_Data")
        qc.to_excel(writer, index=False, sheet_name="QC_Summary")
        clean.to_excel(writer, index=False, sheet_name="Clean_Data_Used")
        outputs["mean_ct"].to_excel(writer, index=False, sheet_name="Mean_Ct")
        outputs["housekeeping_summary"].to_excel(writer, index=False, sheet_name="Housekeeping")
        outputs["delta_ct"].to_excel(writer, index=False, sheet_name="Delta_Ct")
        outputs["final_results"].to_excel(writer, index=False, sheet_name="Fold_Change")

        workbook = writer.book
        number_fmt = workbook.add_format({"num_format": "0.000"})
        header_fmt = workbook.add_format({"bold": True})
        for sheet_name, sheet in writer.sheets.items():
            sheet.set_row(0, None, header_fmt)
            sheet.set_column(0, 30, 18, number_fmt)

    return buffer.getvalue()


def app():
    st.set_page_config(page_title="qPCR Studio", layout="wide")

    st.title("qPCR Studio")
    st.caption("Upload cleaned Ct data → QC → ΔCt → ΔΔCt → fold change → plots → Excel report.")

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

        with st.sidebar:
            default_control_idx = samples.index("Control") if "Control" in samples else 0
            control_sample = st.selectbox("Control/reference sample", samples, index=default_control_idx)

            default_hk = [g for g in ["RPLP0", "TBP", "PPIA", "HPRT", "PBGD"] if g in genes]
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

        top1, top2, top3, top4 = st.columns(4)
        top1.metric("Rows uploaded", len(raw))
        top2.metric("Rows used", len(clean))
        top3.metric("Sample/gene groups", len(qc))
        top4.metric("Outlier values removed", int(qc["n_excluded"].sum()) if apply_qc else 0)

        tabs = st.tabs(["Results", "Plots", "QC", "Raw/Clean data", "Export"])

        with tabs[0]:
            st.subheader("Fold-change results")
            st.dataframe(final, use_container_width=True)

            st.subheader("Housekeeping summary")
            st.dataframe(outputs["housekeeping_summary"], use_container_width=True)

        with tabs[1]:
            st.subheader("Fold-change plots")
            plot_df = final.copy()
            selected_genes = st.multiselect(
                "Genes to plot",
                sorted(plot_df["Gene"].unique()),
                default=sorted(plot_df["Gene"].unique())[: min(4, plot_df["Gene"].nunique())],
            )
            plot_df = plot_df[plot_df["Gene"].isin(selected_genes)]

            if not plot_df.empty:
                fig = px.bar(
                    plot_df,
                    x="Sample",
                    y="fold_change",
                    color="Gene",
                    barmode="group",
                    hover_data=["delta_ct", "delta_delta_ct", "mean_ct"],
                    title="Fold change by sample and gene",
                )
                fig.add_hline(y=1, line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.scatter(
                    plot_df,
                    x="delta_ct",
                    y="fold_change",
                    color="Gene",
                    symbol="Sample",
                    hover_data=["Sample", "Gene", "mean_ct"],
                    title="ΔCt vs fold change",
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Select at least one gene to plot.")

        with tabs[2]:
            st.subheader("Replicate QC")
            st.dataframe(qc, use_container_width=True)

            flagged = qc[qc["qc_status"] != "PASS"]
            if not flagged.empty:
                st.warning("Some groups had outliers, single replicates, or failed QC.")
                st.dataframe(flagged, use_container_width=True)
            else:
                st.success("All replicate groups passed without outlier removal.")

        with tabs[3]:
            left, right = st.columns(2)
            with left:
                st.subheader("Raw data")
                st.dataframe(raw, use_container_width=True)
            with right:
                st.subheader("Clean data used")
                st.dataframe(clean, use_container_width=True)

        with tabs[4]:
            st.subheader("Download report")
            excel_bytes = make_excel_report(raw, qc, clean, outputs)
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
