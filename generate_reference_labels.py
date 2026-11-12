"""
generate_reference_labels.py
=============================
Generates independently reviewed reference QC labels for the synthetic well dataset.
Conforms strictly to REQUIRED_LABEL_COLUMNS:
['well_name', 'depth_top', 'depth_base', 'label', 'reviewer', 'date', 'notes']

Labels:
 0: Acceptable
 1: Questionable
 2: Unreliable
-1: Unreviewed
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
LABELS_DIR = ROOT / "data" / "labels"
(LABELS_DIR / "training").mkdir(parents=True, exist_ok=True)
(LABELS_DIR / "validation").mkdir(parents=True, exist_ok=True)
(LABELS_DIR / "test").mkdir(parents=True, exist_ok=True)

# 1. WELL_A (Training - Clean baseline)
well_a_records = [
    {"well_name": "WELL_A", "depth_top": 8000.0, "depth_base": 11000.0, "label": 0,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Clean baseline log, continuous readings across all tracks"}
]

# 2. WELL_B (Training - Washout + Spikes)
well_b_records = [
    {"well_name": "WELL_B", "depth_top": 8000.0, "depth_base": 8800.0, "label": 0,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Acceptable upper section"},
    {"well_name": "WELL_B", "depth_top": 8800.0, "depth_base": 9000.0, "label": 2,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Severe borehole washout (CALI - BS > 2 in); pad tools likely affected"},
    {"well_name": "WELL_B", "depth_top": 9000.0, "depth_base": 11000.0, "label": 0,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Acceptable lower section with minor isolated spikes"}
]

# 3. WELL_C (Training - Stuck Sensor, missing DT/PEF)
well_c_records = [
    {"well_name": "WELL_C", "depth_top": 8000.0, "depth_base": 9400.0, "label": 0,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Acceptable log, limited curve suite (no DT/PEF)"},
    {"well_name": "WELL_C", "depth_top": 9400.0, "depth_base": 9420.0, "label": 2,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Stuck gamma ray sensor (constant value > 20 ft)"},
    {"well_name": "WELL_C", "depth_top": 9420.0, "depth_base": 11000.0, "label": 0,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Acceptable lower interval"}
]

# 4. WELL_D (Validation - Tight hole + RHOB null gap)
well_d_records = [
    {"well_name": "WELL_D", "depth_top": 8000.0, "depth_base": 9200.0, "label": 0,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Acceptable upper interval"},
    {"well_name": "WELL_D", "depth_top": 9200.0, "depth_base": 9250.0, "label": 1,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Tight hole interval (CALI < bit size); potential tool drag/stick risk"},
    {"well_name": "WELL_D", "depth_top": 9250.0, "depth_base": 9600.0, "label": 0,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Acceptable intermediate interval"},
    {"well_name": "WELL_D", "depth_top": 9600.0, "depth_base": 9620.0, "label": 2,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Density tool signal dropout / null data gap"},
    {"well_name": "WELL_D", "depth_top": 9620.0, "depth_base": 11000.0, "label": 0,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Acceptable lower interval"}
]

# 5. WELL_E (Held-Out Test Well - Washout 9050-9200 ft)
well_e_records = [
    {"well_name": "WELL_E", "depth_top": 8000.0, "depth_base": 9050.0, "label": 0,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Acceptable log"},
    {"well_name": "WELL_E", "depth_top": 9050.0, "depth_base": 9200.0, "label": 2,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Significant borehole enlargement (washout)"},
    {"well_name": "WELL_E", "depth_top": 9200.0, "depth_base": 11000.0, "label": 0,
     "reviewer": "P.Engineer", "date": "2026-10-14", "notes": "Acceptable lower interval"}
]

def save(records, path):
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} intervals to {path}")

save(well_a_records, LABELS_DIR / "training" / "WELL_A_labels.csv")
save(well_b_records, LABELS_DIR / "training" / "WELL_B_labels.csv")
save(well_c_records, LABELS_DIR / "training" / "WELL_C_labels.csv")
save(well_d_records, LABELS_DIR / "validation" / "WELL_D_labels.csv")
save(well_e_records, LABELS_DIR / "test" / "WELL_E_labels.csv")
print("All reference label files successfully generated.")
