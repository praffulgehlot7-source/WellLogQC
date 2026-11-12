"""
welllogqc.reporting
===================
Consolidated HTML and PDF report generation for well log QC and characterisation.
"""

from welllogqc.reporting.report_builder import (
    build_html_report,
    build_pdf_report,
    build_petro_table,
    build_qc_summary_table,
)

__all__ = [
    'build_html_report',
    'build_pdf_report',
    'build_petro_table',
    'build_qc_summary_table',
]
