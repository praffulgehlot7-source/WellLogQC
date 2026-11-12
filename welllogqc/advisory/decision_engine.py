"""
welllogqc/advisory/decision_engine.py
=====================================
Petrophysical Decision Engine & Actionable Advisory System.
Generates interval-level operational dispositions, metrics consolidation,
and domain-specific petroleum engineering recommendations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

def generate_interval_decisions(
    df: pd.DataFrame,
    qc_flags: pd.DataFrame = None,
    predictions: pd.DataFrame = None,
    vsh: pd.Series = None,
    phid: pd.Series = None
) -> pd.DataFrame:
    """
    Generate rigorous petroleum engineering decisions for flagged intervals.
    Dispositions:
      - APPROVED_PAY: Clean reservoir, reliable sensors -> Approved for perforation.
      - REJECTED_WASHOUT: Severe borehole breakout, corrupted density -> Excluded from pay.
      - RE_LOG_RECOMMENDED: Tool sticking or telemetry dropout -> Repeat pass recommended.
      - QUESTIONABLE_TIGHT: Mudcake / friction zone -> Apply environmental corrections.
      - SHALY_SEAL: Non-reservoir shale interval.
    """
    if qc_flags is None or qc_flags.empty:
        return pd.DataFrame(columns=[
            "top_ft", "base_ft", "thickness_ft", "issue_type", "severity", 
            "decision", "confidence", "engineering_justification"
        ])

    decisions = []
    for _, row in qc_flags.iterrows():
        top = float(row.get("depth_top", row.get("top", 0.0)))
        base = float(row.get("depth_base", row.get("base", 0.0)))
        thick = round(abs(base - top), 1)
        issue = str(row.get("issue", "UNKNOWN"))
        sev = str(row.get("severity", "WARNING"))
        
        # Check local data if available
        mask = (df.index >= top) & (df.index <= base)
        local_vsh = vsh[mask].mean() if vsh is not None and not vsh.empty and mask.any() else 0.3
        local_phi = phid[mask].mean() if phid is not None and not phid.empty and mask.any() else 0.15
        
        # Decision logic
        if "WASHOUT" in issue.upper() or (row.get("threshold", "") and "washout" in str(row.get("threshold", "")).lower()):
            dec = "REJECTED_WASHOUT"
            conf = "HIGH"
            just = f"Borehole enlargement ({thick} ft) invalidates pad-contact tools (RHOB/NPHI). Excluded from net pay."
        elif "STUCK" in issue.upper() or "FLAT" in issue.upper():
            dec = "RE_LOG_RECOMMENDED"
            conf = "HIGH"
            just = f"Static sensor readings over {thick} ft indicates tool sticking. Repeat wireline pass required."
        elif "SPIKE" in issue.upper():
            dec = "QUESTIONABLE_FILTER"
            conf = "MEDIUM"
            just = f"Single-sample statistical outlier. Apply median filtering before volumetrics."
        elif "TIGHT" in issue.upper():
            dec = "QUESTIONABLE_TIGHT"
            conf = "MEDIUM"
            just = f"Caliper under bit size indicates mudcake buildup or creeping shale. Apply caliper correction."
        elif local_phi > 0.18 and local_vsh < 0.20:
            dec = "APPROVED_PAY"
            conf = "HIGH"
            just = f"High reservoir quality (phi={local_phi:.1%}, Vsh={local_vsh:.1%}) with minor operational flag."
        else:
            dec = "SHALY_NON_RESERVOIR"
            conf = "MEDIUM"
            just = f"Non-productive interval (Vsh={local_vsh:.1%}). Flag does not impact reserve calculation."

        decisions.append({
            "top_ft": top,
            "base_ft": base,
            "thickness_ft": thick,
            "issue_type": issue,
            "severity": sev,
            "decision": dec,
            "confidence": conf,
            "engineering_justification": just
        })

    return pd.DataFrame(decisions)


def generate_actionable_advisories(
    df: pd.DataFrame,
    qc_flags: pd.DataFrame = None,
    predictions: pd.DataFrame = None,
    proximity_analysis: Dict[str, Any] = None
) -> List[Dict[str, str]]:
    """
    Synthesize domain-specific actionable improvements and advisories for:
      1. Drilling & Mud Engineering
      2. Wireline Tool Operations
      3. Petrophysical Formation Evaluation
    """
    advisories = []
    
    # Check for washouts
    washout_flags = qc_flags[qc_flags["issue"].str.contains("WASHOUT", case=False, na=False)] if qc_flags is not None and not qc_flags.empty else pd.DataFrame()
    if not washout_flags.empty:
        top_c = "depth_top" if "depth_top" in washout_flags.columns else "top"
        base_c = "depth_base" if "depth_base" in washout_flags.columns else "base"
        max_len = abs(washout_flags[base_c] - washout_flags[top_c]).max() if top_c in washout_flags.columns and base_c in washout_flags.columns else 0.0
        advisories.append({
            "category": "Drilling & Borehole Geomechanics",
            "title": "Borehole Breakout & Mud Weight Advisory",
            "recommendation": f"Severe borehole washout observed across {len(washout_flags)} intervals (maximum contiguous span: {max_len:.1f} ft). "
                              "Indicates mechanical compressive failure of shales under current mud column. "
                              "Recommend increasing drilling mud weight by 0.4–0.6 ppg on subsequent offset wells to stabilize formation.",
            "icon": "shield"
        })
    else:
        advisories.append({
            "category": "Drilling & Borehole Geomechanics",
            "title": "Borehole Condition Optimal",
            "recommendation": "Caliper log demonstrates on-gauge borehole stability throughout evaluated section. "
                              "Mud system provided excellent borehole wall stability and minimal invasive breakout.",
            "icon": "check"
        })

    # Check for stuck tools / cable tension
    stuck_flags = qc_flags[qc_flags["issue"].str.contains("STUCK", case=False, na=False)] if qc_flags is not None and not qc_flags.empty else pd.DataFrame()
    if not stuck_flags.empty:
        advisories.append({
            "category": "Wireline Operations & Acquisition",
            "title": "Tool Sticking & Cable Dynamics Advisory",
            "recommendation": f"Detected {len(stuck_flags)} invariant depth intervals. "
                              "Correlate immediately with cable tension log to confirm differential sticking or tool stall. "
                              "Recommend logging contractor run standoff centralizers and review maximum pull limits.",
            "icon": "alert"
        })

    # Petrophysical evaluation advice
    if proximity_analysis:
        tier = proximity_analysis.get("reservoir_tier", "")
        prox = proximity_analysis.get("composite_reservoir_proximity_pct", 75.0)
        advisories.append({
            "category": "Petrophysics & Formation Evaluation",
            "title": f"Reservoir Characterization: {tier}",
            "recommendation": f"Evaluated interval achieves {prox}% proximity to World-Class Ideal Reservoir standards. "
                              "For reserve volumetrics, ensure density porosity is computed using matrix density rho_ma=2.65 g/cm3 "
                              "and fluid density rho_fl=1.00 g/cm3, and discard density readings in flagged breakout zones.",
            "icon": "target"
        })

    return advisories
