"""
welllogqc.reporting.report_builder
==================================
HTML and PDF report generator integrating ingestion, rule-based QC,
petrophysics, NTG, composite plots, and ML predictions.
"""

import base64
import datetime
import html
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from jinja2 import Environment, FileSystemLoader
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_qc_summary_table(qc_flags: pd.DataFrame) -> str:
    """Construct an HTML summary table of QC flags colour-coded by severity.

    Row colours match standard severity styling:
    - CRITICAL: Red (#f8d7da / .table-danger)
    - WARNING: Orange/Yellow (#fff3cd / .table-warning)
    - INFO: Blue (#cff4fc / .table-info)

    Args:
        qc_flags: DataFrame of QC flags complying with standard QC_COLS.

    Returns:
        str: HTML table markup string.
    """
    if qc_flags is None or qc_flags.empty:
        return (
            '<div class="alert alert-success mb-0 py-2">'
            'No QC issues or anomalous intervals detected. All evaluated curves passed checks.'
            '</div>'
        )

    rows_html: list[str] = []
    for row in qc_flags.itertuples(index=False):
        sev = str(getattr(row, 'severity', 'INFO')).upper().strip()
        if sev == 'CRITICAL':
            row_class = 'table-danger row-critical'
            badge_class = 'bg-danger'
        elif sev == 'WARNING':
            row_class = 'table-warning row-warning'
            badge_class = 'bg-warning text-dark'
        else:
            row_class = 'table-info row-info'
            badge_class = 'bg-info text-dark'

        d_top = getattr(row, 'depth_top', np.nan)
        d_base = getattr(row, 'depth_base', np.nan)
        top_str = f"{float(d_top):.1f}" if pd.notna(d_top) else "-"
        base_str = f"{float(d_base):.1f}" if pd.notna(d_base) else "-"

        curve_str = html.escape(str(getattr(row, 'curve', '-')))
        issue_str = html.escape(str(getattr(row, 'issue', '-')))
        method_str = html.escape(str(getattr(row, 'method', '-')))
        val_raw = getattr(row, 'value', '-')
        thresh_raw = getattr(row, 'threshold', '-')

        try:
            val_str = f"{float(val_raw):.2f}" if pd.notna(val_raw) and isinstance(val_raw, (int, float)) else html.escape(str(val_raw))
        except (ValueError, TypeError):
            val_str = html.escape(str(val_raw))

        thresh_str = html.escape(str(thresh_raw))
        desc_str = html.escape(str(getattr(row, 'description', '')))

        rows_html.append(
            f'<tr class="{row_class}">'
            f'<td>{top_str}</td>'
            f'<td>{base_str}</td>'
            f'<td><strong>{curve_str}</strong></td>'
            f'<td>{issue_str}</td>'
            f'<td><span class="badge {badge_class}">{sev}</span></td>'
            f'<td>{method_str}</td>'
            f'<td>{val_str}</td>'
            f'<td>{thresh_str}</td>'
            f'<td>{desc_str}</td>'
            f'</tr>'
        )

    tbody = "\n".join(rows_html)
    return (
        '<table class="table table-bordered table-sm table-hover align-middle mb-0">\n'
        '  <thead class="table-light">\n'
        '    <tr>\n'
        '      <th>Top (ft)</th>\n'
        '      <th>Base (ft)</th>\n'
        '      <th>Curve</th>\n'
        '      <th>Issue</th>\n'
        '      <th>Severity</th>\n'
        '      <th>Method</th>\n'
        '      <th>Value</th>\n'
        '      <th>Threshold</th>\n'
        '      <th>Description</th>\n'
        '    </tr>\n'
        '  </thead>\n'
        f'  <tbody>\n{tbody}\n  </tbody>\n'
        '</table>'
    )


def build_petro_table(petro_df: Optional[pd.DataFrame]) -> str:
    """Format petrophysical calculation curves as a sample HTML table.

    Downsamples the dataset to at most 50 evenly spaced rows so the rendered
    report remains readable and lightweight.

    Args:
        petro_df: DataFrame containing petrophysical evaluation curves
            (e.g. VSH, PHID, SW), indexed by DEPT or with a DEPT column.

    Returns:
        str: HTML table markup string.
    """
    if petro_df is None or petro_df.empty:
        return '<p class="text-muted fst-italic mb-0">No petrophysical evaluation curves available.</p>'

    df = petro_df.copy()
    if 'DEPT' not in df.columns:
        df['DEPT'] = df.index.to_numpy(dtype=np.float64)

    total_rows = len(df)
    max_rows = 50
    if total_rows > max_rows:
        step = int(np.ceil(total_rows / max_rows))
        sampled = df.iloc[::step].copy()
    else:
        sampled = df.copy()

    # Place DEPT as first column
    cols = ['DEPT'] + [c for c in sampled.columns if c != 'DEPT']
    sampled = sampled[cols]

    # Format numeric values
    for c in cols:
        if pd.api.types.is_numeric_dtype(sampled[c]):
            sampled[c] = sampled[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")

    table_html = sampled.to_html(
        classes='table table-striped table-bordered table-sm text-center align-middle mb-0',
        index=False,
        escape=False
    )
    caption = (
        f'<p class="text-muted small mt-1 mb-0">'
        f'Displaying {len(sampled)} sampled depths out of {total_rows} total rows.'
        f'</p>'
    )
    return table_html + caption


def _format_ingestion_summary(ingestion_report: Any) -> str:
    """Render the ingestion report as an HTML summary table."""
    if not ingestion_report:
        return '<p class="text-muted fst-italic mb-0">No ingestion metadata provided.</p>'

    if isinstance(ingestion_report, str):
        return ingestion_report

    if isinstance(ingestion_report, dict):
        rows = []
        for k, v in ingestion_report.items():
            key_label = html.escape(str(k).replace('_', ' ').title())
            if isinstance(v, (list, tuple, set)):
                val_str = html.escape(", ".join(str(item) for item in v))
            elif isinstance(v, float):
                val_str = f"{v:.2f}"
            else:
                val_str = html.escape(str(v))
            rows.append(f"<tr><th class='w-25 text-nowrap'>{key_label}</th><td>{val_str}</td></tr>")
        return (
            '<table class="table table-sm table-bordered mb-0 bg-white">'
            f"<tbody>{''.join(rows)}</tbody>"
            '</table>'
        )

    return f"<pre class='mb-0'>{html.escape(str(ingestion_report))}</pre>"


def _format_ntg_summary(ntg_result: Optional[Dict[str, Any]]) -> str:
    """Render Net-to-Gross metrics into Bootstrap cards or summary table."""
    if not ntg_result:
        return '<p class="text-muted fst-italic mb-0">No Net-to-Gross (NTG) calculations available.</p>'

    items = []
    for k, v in ntg_result.items():
        title = html.escape(str(k).replace('_', ' ').upper())
        if isinstance(v, float):
            val_str = f"{v:.3f}" if abs(v) < 1.0 else f"{v:.1f}"
        else:
            val_str = html.escape(str(v))
        items.append(
            f'<div class="col-md-3 col-sm-6 mb-2">'
            f'  <div class="card p-2 text-center border-0 bg-white shadow-sm">'
            f'    <div class="text-muted small fw-bold">{title}</div>'
            f'    <div class="fs-5 fw-bold text-primary">{val_str}</div>'
            f'  </div>'
            f'</div>'
        )
    return f'<div class="row g-2">{"".join(items)}</div>'


def _format_ml_section(predictions: Optional[pd.DataFrame]) -> str:
    """Render ML predictions summary statistics and high-risk flags."""
    if predictions is None or predictions.empty:
        return '<p class="text-muted fst-italic mb-0">No ML risk predictions available for this well.</p>'

    total_pts = len(predictions)
    score_col = 'QC_RISK_SCORE'
    label_col = 'QC_RISK_LABEL'

    flagged_pts = int((predictions[label_col] > 0).sum()) if label_col in predictions.columns else 0
    pct_flagged = (flagged_pts / total_pts * 100) if total_pts > 0 else 0.0

    mean_score = float(predictions[score_col].mean()) if score_col in predictions.columns else 0.0
    max_score = float(predictions[score_col].max()) if score_col in predictions.columns else 0.0

    status_badge = (
        '<span class="badge bg-danger fs-6">Elevated QC Risk</span>'
        if pct_flagged > 10.0 or mean_score > 0.4
        else '<span class="badge bg-success fs-6">Normal Quality Profile</span>'
    )

    summary_cards = (
        f'<div class="row g-2 mb-3">'
        f'  <div class="col-md-3 col-sm-6">'
        f'    <div class="card p-2 text-center border-0 bg-white shadow-sm">'
        f'      <div class="text-muted small fw-bold">TOTAL DEPTH SAMPLES</div>'
        f'      <div class="fs-5 fw-bold">{total_pts:,}</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="col-md-3 col-sm-6">'
        f'    <div class="card p-2 text-center border-0 bg-white shadow-sm">'
        f'      <div class="text-muted small fw-bold">FLAGGED SAMPLES</div>'
        f'      <div class="fs-5 fw-bold text-danger">{flagged_pts:,} ({pct_flagged:.1f}%)</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="col-md-3 col-sm-6">'
        f'    <div class="card p-2 text-center border-0 bg-white shadow-sm">'
        f'      <div class="text-muted small fw-bold">MEAN RISK SCORE</div>'
        f'      <div class="fs-5 fw-bold text-primary">{mean_score:.3f}</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="col-md-3 col-sm-6">'
        f'    <div class="card p-2 text-center border-0 bg-white shadow-sm">'
        f'      <div class="text-muted small fw-bold">MAX RISK SCORE</div>'
        f'      <div class="fs-5 fw-bold text-danger">{max_score:.3f}</div>'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )

    return f'<div class="mb-2 d-flex justify-content-between align-items-center"><strong>Model Risk Assessment:</strong> {status_badge}</div>{summary_cards}'


def build_html_report(
    well_name: str,
    ingestion_report: Any,
    qc_flags: pd.DataFrame,
    petro_df: Optional[pd.DataFrame],
    ntg_result: Optional[Dict[str, Any]],
    predictions: Optional[pd.DataFrame],
    plot_path: Optional[Union[str, Path]],
    output_path: Union[str, Path]
) -> str:
    """Render consolidated HTML report using Jinja2 template and save to disk.

    Embeds any log plot as a base64 encoded image for complete self-containment.

    Args:
        well_name: Identifier name of the well.
        ingestion_report: Ingestion metadata dictionary or report summary.
        qc_flags: DataFrame of rule-based QC check flags.
        petro_df: Optional DataFrame of petrophysical curves (Vsh, phiD, Sw).
        ntg_result: Optional dictionary of reservoir Net-to-Gross metrics.
        predictions: Optional DataFrame containing ML risk scores and labels.
        plot_path: Optional file path to a composite log plot PNG image.
        output_path: File system path where the HTML report will be written.

    Returns:
        str: Complete rendered HTML content.
    """
    plot_b64 = ""
    if plot_path is not None:
        p_plot = Path(plot_path)
        if p_plot.is_file():
            try:
                with open(p_plot, 'rb') as f_img:
                    plot_b64 = base64.b64encode(f_img.read()).decode('utf-8')
            except Exception as exc:
                logger.warning(f"Could not encode plot image at {p_plot}: {exc}")

    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ingestion_html = _format_ingestion_summary(ingestion_report)
    flags_table_html = build_qc_summary_table(qc_flags)
    petro_table_html = build_petro_table(petro_df)
    ntg_summary_html = _format_ntg_summary(ntg_result)
    ml_section_html = _format_ml_section(predictions)

    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)
    template = env.get_template("report.html")

    html_content = template.render(
        well_name=well_name,
        report_date=report_date,
        ingestion=ingestion_html,
        flags_table=flags_table_html,
        petro_table=petro_table_html,
        ntg_summary=ntg_summary_html,
        plot_b64=plot_b64,
        ml_section=ml_section_html,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding='utf-8')

    return html_content


def build_pdf_report(
    html_path: Union[str, Path],
    output_path: Union[str, Path]
) -> bool:
    """Convert an existing HTML report to a PDF document via weasyprint.

    Imports weasyprint dynamically inside the function so that absent system
    cairo/pango libraries do not prevent the rest of the application from running.

    Args:
        html_path: Path to the input HTML report.
        output_path: Path to write the output PDF document.

    Returns:
        bool: True if PDF generation succeeded, False otherwise.
    """
    h_file = Path(html_path)
    if not h_file.is_file():
        logger.warning(f"HTML source file for PDF generation not found: {h_file}")
        return False

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        import weasyprint
        weasyprint.HTML(filename=str(h_file)).write_pdf(str(out_file))
        logger.info(f"Successfully generated PDF report at: {out_file}")
        return True
    except Exception as exc:
        logger.warning(f"weasyprint PDF conversion failed for {h_file}: {exc}")
        return False
