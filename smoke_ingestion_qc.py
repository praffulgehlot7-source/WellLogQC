"""
smoke_ingestion_qc.py
---------------------
Smoke test for ingestion and QC pipeline using sample LAS files.
"""

from welllogqc.ingestion.las_reader import LASReader
from welllogqc.qc.qc_reporter import run_all_qc, summarise_flags

reader = LASReader()

# Test WELL_A (full curves)
result = reader.read('data/raw/training/WELL_A.las', role='training')
print(f'WELL_A: {result.well_name}, curves: {result.available_curves}')
print(f'  missing: {result.missing_curves}')
print(f'  depth: {result.depth_start}-{result.depth_stop} ft, step={result.depth_step}')
flags = run_all_qc(result)
print(f'  QC flags: {len(flags)}')
print(flags[['depth_top','curve','issue','severity','description']].head(5).to_string())

# Test WELL_C (missing curves)
result_c = reader.read('data/raw/training/WELL_C.las', role='training')
print(f'WELL_C missing curves: {result_c.missing_curves}')
flags_c = run_all_qc(result_c)
print(f'WELL_C QC flags: {len(flags_c)}')
print(summarise_flags(flags_c))

# Test WELL_B (spikes + washout)
result_b = reader.read('data/raw/training/WELL_B.las', role='training')
flags_b = run_all_qc(result_b)
print(f'WELL_B flags: {len(flags_b)}')
print(flags_b.groupby('issue').size())
