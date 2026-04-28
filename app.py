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
    exclude_failed_groups: bool


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make input column names predictable while preserving only needed fields."""
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
    return pd.DataFrame(rows)


def replicate_qc(raw: pd.DataFrame, settings: AnalysisSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = raw.groupby(["Sample", "Gene"], as_index=False).agg(
        n_replicates=("Ct", "count"),
        min_ct=("Ct", "min"),
        max_ct=("Ct", "max"),
        mean_ct_all=("Ct", "mean"),
        sd_ct_all=("Ct", "std"),
    )
    grouped["ct_spread"] = grouped["max_ct"] - grouped["min_ct"]
    grouped["qc_status"] = np.where(
        grouped["ct_spread"] > settings.outlier_cutoff,
        "FAIL_spread_over_cutoff",
        "PASS",
    )

    if settings.exclude_failed_groups:
        failed_keys = grouped.loc[grouped["qc_status"] != "PASS", ["Sample", "Gene"]]
        if len(failed_keys) > 0:
            key_tuples = set(map(tuple, failed_keys.to_numpy()))
            mask = raw[["Sample", "Gene"]].apply(tuple, axis=1).isin(key_tuples)
            clean = raw.loc[~mask].copy()
        else:
            clean = raw.copy()
    else:
        clean = raw.copy()

    return grouped, clean


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
            "Some samples are missing housekeeping data: "
            + ", ".join(map(str, missing_hk))
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

    # Keep a clean display order.
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
        for sheet in writer.sheets.values():
            sheet.set_column(0, 20, 18, number_fmt)

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
                "Fail replicate group if Ct spread is greater than",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
            )
            exclude_failed_groups = st.checkbox(
                "Exclude failed replicate groups from analysis",
                value=True,
                help="If checked, an entire sample/gene group is removed when max Ct - min Ct exceeds the cutoff.",
            )

        settings = AnalysisSettings(
            control_sample=control_sample,
            housekeeping_genes=housekeeping_genes,
            outlier_cutoff=float(outlier_cutoff),
            exclude_failed_groups=exclude_failed_groups,
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
        top4.metric("QC failed groups", int((qc["qc_status"] != "PASS").sum()))

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
            failed = qc[qc["qc_status"] != "PASS"]
            if not failed.empty:
                st.warning("Some replicate groups failed the Ct spread cutoff.")
                st.dataframe(failed, use_container_width=True)
            else:
                st.success("All replicate groups passed the current spread cutoff.")

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
