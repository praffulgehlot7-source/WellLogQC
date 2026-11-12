"""
ingestion_summary.py
--------------------
Formats ingestion diagnostics into human-readable textual reports and
JSON-serialisable dictionaries for downstream consumers and dashboards.
"""

from typing import Any
from welllogqc.ingestion.las_reader import IngestionResult


def format_ingestion_report(result: IngestionResult) -> str:
    """Formats an IngestionResult into a detailed multi-line human-readable summary.

    Args:
        result: IngestionResult instance containing ingested well data and metadata.

    Returns:
        Formatted multi-line report string.

    Raises:
        ValueError: If result is not an IngestionResult instance.
    """
    if not isinstance(result, IngestionResult):
        raise ValueError(f"Expected IngestionResult, got {type(result).__name__}")

    total_samples = len(result.df)
    span = abs(result.depth_stop - result.depth_start)

    lines: list[str] = [
        "=" * 78,
        f"WELL LOG INGESTION REPORT: {result.well_name}",
        "=" * 78,
        f"Role:                 {result.role}",
        f"Source LAS File:      {result.las_path}",
        f"Depth Interval:       {result.depth_start:.2f} - {result.depth_stop:.2f} ft (Span: {span:.2f} ft)",
        f"Depth Step:           {result.depth_step:.4f} ft",
        f"Total Samples:        {total_samples:,}",
        f"Header Null Sentinel: {result.null_value}",
        "",
        "CURVE AVAILABILITY:",
        f"  Available ({len(result.available_curves)}):   {', '.join(result.available_curves) if result.available_curves else 'None'}",
        f"  Missing ({len(result.missing_curves)}):     {', '.join(result.missing_curves) if result.missing_curves else 'None'}",
        f"  Unknown ({len(result.unknown_curves)}):     {', '.join(result.unknown_curves) if result.unknown_curves else 'None'}",
        "",
        "MNEMONIC MAPPINGS:",
    ]

    for raw, canonical in sorted(result.curve_map.items()):
        lines.append(f"  {raw:<12} -> {canonical}")

    lines.append("")
    lines.append("DATA COMPLETENESS / NULL STATISTICS:")
    lines.append(f"  {'Curve':<10} {'Valid':<10} {'Nulls':<10} {'Null %':<8}")
    lines.append("  " + "-" * 38)

    for curve in result.available_curves:
        null_count = result.null_counts.get(curve, 0)
        valid_count = total_samples - null_count
        null_pct = (null_count / total_samples * 100.0) if total_samples > 0 else 0.0
        lines.append(f"  {curve:<10} {valid_count:<10,} {null_count:<10,} {null_pct:>6.1f}%")

    lines.append("")
    lines.append(f"INGESTION WARNINGS ({len(result.warnings)}):")
    if result.warnings:
        for w in result.warnings:
            lines.append(f"  [!] {w}")
    else:
        lines.append("  None. Clean ingestion.")

    lines.append("=" * 78)
    return "\n".join(lines)


def ingestion_to_dict(result: IngestionResult) -> dict[str, Any]:
    """Converts an IngestionResult into a JSON-serialisable dictionary.

    The returned dictionary excludes the underlying DataFrame to keep the
    payload lightweight for UI state management.

    Args:
        result: IngestionResult instance to serialise.

    Returns:
        JSON-serialisable dictionary of ingestion metadata.

    Raises:
        ValueError: If result is not an IngestionResult instance.
    """
    if not isinstance(result, IngestionResult):
        raise ValueError(f"Expected IngestionResult, got {type(result).__name__}")

    return {
        "well_name": str(result.well_name),
        "role": str(result.role),
        "las_path": str(result.las_path),
        "depth_start": float(result.depth_start),
        "depth_stop": float(result.depth_stop),
        "depth_step": float(result.depth_step),
        "total_samples": int(len(result.df)),
        "null_value": float(result.null_value),
        "available_curves": list(result.available_curves),
        "missing_curves": list(result.missing_curves),
        "unknown_curves": list(result.unknown_curves),
        "curve_map": {str(k): str(v) for k, v in result.curve_map.items()},
        "null_counts": {str(k): int(v) for k, v in result.null_counts.items()},
        "warnings": list(result.warnings),
    }
