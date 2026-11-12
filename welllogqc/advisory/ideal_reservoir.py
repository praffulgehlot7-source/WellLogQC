"""
welllogqc/advisory/ideal_reservoir.py
====================================
Petrophysical Ideal Reservoir Benchmark System.
Provides standard ideal petrophysical reference baselines, gap analysis,
and multi-parameter proximity scoring.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

# Ideal Reservoir Reference Parameters (World-Class Clean Sandstone Pay)
IDEAL_RESERVOIR_BENCHMARKS = {
    "reservoir_type": "Prolific Clean Clastic Sandstone (e.g. Wilcox / Brent)",
    "gamma_ray_gapi": {"ideal": 25.0, "min_acceptable": 10.0, "max_acceptable": 45.0},
    "vsh_fraction": {"ideal": 0.08, "cutoff": 0.15},
    "caliper_gauge_dev_in": {"ideal": -0.15, "min_acceptable": -0.5, "max_acceptable": 0.3},
    "porosity_fraction": {"ideal": 0.24, "cutoff": 0.15, "world_class": 0.28},
    "water_saturation_fraction": {"ideal": 0.20, "cutoff": 0.40},
    "deep_resistivity_ohmm": {"ideal": 85.0, "cutoff": 20.0},
    "acoustic_transit_time_us_ft": {"ideal": 74.0, "range": [68.0, 85.0]},
    "net_to_gross": {"ideal": 0.90, "cutoff": 0.65}
}

def compute_ideal_reservoir_gap_analysis(
    df: pd.DataFrame,
    vsh: pd.Series = None,
    phid: pd.Series = None,
    sw: pd.Series = None
) -> Dict[str, Any]:
    """
    Compare well petrophysical characteristics against the Ideal Reservoir Benchmark.
    Returns composite indices (0-100%) and parameter deviations.
    """
    results = {}
    
    # 1. Borehole Gauge Stability Index
    if "CALI" in df.columns and "BS" in df.columns:
        diff = (df["CALI"] - df["BS"]).abs()
        mean_diff = float(diff.mean())
        # Gauge stability: 100% if mean_diff <= 0.2 in, decays linearly
        gauge_index = max(0.0, min(100.0, (1.0 - (mean_diff / 2.0)) * 100.0))
        results["gauge_stability_index"] = round(gauge_index, 1)
        results["mean_caliper_deviation_in"] = round(mean_diff, 2)
    else:
        results["gauge_stability_index"] = 85.0
        results["mean_caliper_deviation_in"] = 0.25

    # 2. Matrix Cleanliness Index (Vsh)
    if vsh is not None and not vsh.empty:
        mean_vsh = float(vsh.mean())
        cleanliness_index = max(0.0, min(100.0, (1.0 - mean_vsh) * 100.0))
        results["cleanliness_index"] = round(cleanliness_index, 1)
        results["mean_vsh"] = round(mean_vsh, 3)
    elif "GR" in df.columns:
        mean_gr = float(df["GR"].mean())
        # GR ideal 25, shale 120
        cleanliness_index = max(0.0, min(100.0, (1.0 - max(0.0, mean_gr - 25.0) / 95.0) * 100.0))
        results["cleanliness_index"] = round(cleanliness_index, 1)
        results["mean_vsh"] = round(max(0.0, (mean_gr - 25.0) / 95.0), 3)
    else:
        results["cleanliness_index"] = 75.0
        results["mean_vsh"] = 0.25

    # 3. Porosity Quality Index
    if phid is not None and not phid.empty:
        mean_phi = float(phid.clip(lower=0.0).mean())
        # 0.24 is ideal 100%
        phi_index = max(0.0, min(100.0, (mean_phi / 0.24) * 100.0))
        results["porosity_quality_index"] = round(phi_index, 1)
        results["mean_porosity"] = round(mean_phi, 3)
    else:
        results["porosity_quality_index"] = 80.0
        results["mean_porosity"] = 0.19

    # 4. Hydrocarbon Saturation Index (1 - Sw)
    if sw is not None and not sw.empty:
        mean_sw = float(sw.clip(lower=0.0, upper=1.0).mean())
        hc_index = max(0.0, min(100.0, (1.0 - mean_sw) * 100.0))
        results["hydrocarbon_saturation_index"] = round(hc_index, 1)
        results["mean_water_saturation"] = round(mean_sw, 3)
    elif "RT" in df.columns:
        mean_rt = float(df["RT"].clip(lower=0.1).mean())
        # RT ideal 85, water zone 1-2
        rt_index = max(0.0, min(100.0, (np.log10(max(1.0, mean_rt)) / np.log10(85.0)) * 100.0))
        results["hydrocarbon_saturation_index"] = round(rt_index, 1)
        results["mean_water_saturation"] = round(max(0.15, 1.0 - (rt_index / 100.0)), 3)
    else:
        results["hydrocarbon_saturation_index"] = 70.0
        results["mean_water_saturation"] = 0.30

    # Record mean GR and mean RT if available
    if "GR" in df.columns:
        results["mean_gr"] = round(float(df["GR"].mean()), 1)
    if "RT" in df.columns:
        results["mean_rt"] = round(float(df["RT"].clip(lower=0.1).mean()), 1)

    # Composite Reservoir Quality Proximity Score (0 - 100)
    composite = (
        0.25 * results["gauge_stability_index"] +
        0.25 * results["cleanliness_index"] +
        0.30 * results["porosity_quality_index"] +
        0.20 * results["hydrocarbon_saturation_index"]
    )
    results["composite_reservoir_proximity_pct"] = round(composite, 1)

    # Classification relative to Ideal Reservoir
    if composite >= 85.0:
        results["reservoir_tier"] = "Tier 1: World-Class Commercial Pay"
    elif composite >= 70.0:
        results["reservoir_tier"] = "Tier 2: Economic Moderate Quality Reservoir"
    elif composite >= 50.0:
        results["reservoir_tier"] = "Tier 3: Marginal / Shaly Sand Resource"
    else:
        results["reservoir_tier"] = "Tier 4: Sub-economic / Tight / Non-reservoir"

    return results

def get_ideal_benchmark_comparison_table(analysis: Dict[str, Any]) -> pd.DataFrame:
    """
    Format a structured side-by-side comparison table comparing
    evaluated well metrics against the World-Class Ideal Clastic Reservoir Standard,
    including a dedicated column with key comparison notes and engineering conclusions.
    """
    mean_dev = analysis.get("mean_caliper_deviation_in", 0.0)
    mean_vsh = analysis.get("mean_vsh", 0.0)
    mean_phi = analysis.get("mean_porosity", 0.0)
    mean_sw = analysis.get("mean_water_saturation", 0.2)
    sh_sat = max(0.0, 1.0 - mean_sw)
    mean_gr = analysis.get("mean_gr")
    mean_rt = analysis.get("mean_rt")

    # 1. Borehole Gauge Notes
    if mean_dev <= 0.25:
        gauge_notes = f"On-gauge hole (+{mean_dev:.2f} in dev). Optimal mud weight prevents shale breakout; pad sensors (RHOB/NPHI) maintain 100% rock contact for reliable porosity evaluation."
    elif mean_dev <= 0.70:
        gauge_notes = f"Moderate hole enlargement (+{mean_dev:.2f} in over bit). Minor shale sloughing; verify density pad standoff and apply geometric borehole cavity corrections."
    else:
        gauge_notes = f"Severe hole washout (+{mean_dev:.2f} in exceeds 0.50 in cutoff). Pad sensors lost wall contact, artificially elevating apparent porosity; recommend sonic porosity substitution and +0.5 ppg mud weight on offset wells."

    # 2. Formation Cleanliness Notes
    if mean_vsh <= 0.12:
        vsh_notes = f"Pristine quartz sand matrix ({mean_vsh:.1%} shale vs <=8.0% ideal). Unclogged pore throats provide optimal permeability and minimal bound water, ensuring high recovery factor."
    elif mean_vsh <= 0.25:
        vsh_notes = f"Moderate clay fraction ({mean_vsh:.1%}). Within commercial pay cutoff (<=15%); minor silty laminations reduce horizontal-to-vertical permeability ratio."
    else:
        vsh_notes = f"Clay-rich formation ({mean_vsh:.1%} exceeds 15% cutoff). Dispersed clay minerals choke pore throats; Archie equation may overestimate water cut due to clay electrical conductivity."

    # 3. Storage Porosity Notes
    if mean_phi >= 0.20:
        phi_notes = f"World-class storage capacity ({mean_phi:.1%} matches >=24% Brent/Wilcox ideal). Represents massive hydrocarbon pore volume per acre-foot with strong commercial drainage potential."
    elif mean_phi >= 0.14:
        phi_notes = f"Commercial reservoir porosity ({mean_phi:.1%} meets >=15% economic cutoff). Adequate pore volume supporting sustained commercial production rates and proven reserve booking."
    else:
        phi_notes = f"Tight rock matrix ({mean_phi:.1%} falls below 15% economic cutoff). Sub-economic conventional pore volume; requires hydraulic fracture stimulation or horizontal completions to flow."

    # 4. Hydrocarbon Saturation Notes
    if sh_sat >= 0.70:
        sw_notes = f"High hydrocarbon cut ({sh_sat:.1%} approaches >=80% ideal pay). Only {mean_sw:.1%} irreducible bound water; minimal risk of water production upon perforation; prime commercial pay target."
    elif sh_sat >= 0.50:
        sw_notes = f"Economic hydrocarbon fraction ({sh_sat:.1%} meets >=60% cutoff). Commercial hydrocarbon pay with potential for minor initial water cut; establish fluid contacts before perforating."
    else:
        sw_notes = f"High water saturation ({mean_sw:.1%}; hydrocarbon cut only {sh_sat:.1%} fails 60% cutoff). Water-dominated pore space; represents depleted transition zone or aquifer; not recommended for completion."

    rows = [
        {
            "Reservoir Parameter": "Borehole Gauge Stability",
            "Given Well Data": f"{mean_dev:.2f} in deviation",
            "Ideal Benchmark Standard": "<= 0.15 in (On-Gauge Bit)",
            "Engineering Tolerance Cutoff": "< 0.50 in (Max Breakout)",
            "Match Proximity": f"{analysis.get('gauge_stability_index', 0):.1f}%",
            "Evaluation Verdict": "Optimal Hole Condition" if mean_dev <= 0.25 else ("Moderate Washout" if mean_dev <= 0.7 else "Severe Breakout"),
            "Key Comparison Notes & Concluding Remarks": gauge_notes
        },
        {
            "Reservoir Parameter": "Formation Cleanliness (Vsh)",
            "Given Well Data": f"{mean_vsh:.1%}",
            "Ideal Benchmark Standard": "<= 8.0% (Clean Quartz Sand)",
            "Engineering Tolerance Cutoff": "<= 15.0% (Shale Cutoff)",
            "Match Proximity": f"{analysis.get('cleanliness_index', 0):.1f}%",
            "Evaluation Verdict": "Clean Reservoir Sand" if mean_vsh <= 0.15 else "Shaly Sand / Non-Pay",
            "Key Comparison Notes & Concluding Remarks": vsh_notes
        }
    ]

    # Optional GR row if GR was logged
    if mean_gr is not None:
        gr_prox = max(0.0, min(100.0, (1.0 - max(0.0, mean_gr - 25.0) / 95.0) * 100.0))
        if mean_gr <= 45.0:
            gr_notes = f"Low natural radioactive emission ({mean_gr:.1f} gAPI). Confirms clean quartz sandstone matrix with minimal potassium/thorium radioactive clay minerals."
            gr_verdict = "Clean Quartz Matrix"
        elif mean_gr <= 75.0:
            gr_notes = f"Intermediate radioactivity ({mean_gr:.1f} gAPI). Indicates shaly sandstone, feldspathic arkose, or siltstone interbeds; evaluate alongside neutron-density crossplots."
            gr_verdict = "Slightly Argillaceous"
        else:
            gr_notes = f"High radioactivity ({mean_gr:.1f} gAPI exceeds 60 gAPI cutoff). Dominated by marine illite/smectite clays or organic-rich shale seal; non-reservoir flow unit."
            gr_verdict = "Shale / Non-Reservoir"

        rows.append({
            "Reservoir Parameter": "Formation Radioactivity (GR)",
            "Given Well Data": f"{mean_gr:.1f} gAPI",
            "Ideal Benchmark Standard": "15.0 - 35.0 gAPI (Clean Quartz)",
            "Engineering Tolerance Cutoff": "<= 60.0 gAPI (Pay Cutoff)",
            "Match Proximity": f"{gr_prox:.1f}%",
            "Evaluation Verdict": gr_verdict,
            "Key Comparison Notes & Concluding Remarks": gr_notes
        })

    # Storage Porosity Row
    rows.append({
        "Reservoir Parameter": "Matrix Storage Porosity (phiD)",
        "Given Well Data": f"{mean_phi:.1%}",
        "Ideal Benchmark Standard": ">= 24.0% (High Porosity Pay)",
        "Engineering Tolerance Cutoff": ">= 15.0% (Economic Cutoff)",
        "Match Proximity": f"{analysis.get('porosity_quality_index', 0):.1f}%",
        "Evaluation Verdict": "Prolific Storage Capacity" if mean_phi >= 0.20 else ("Commercial Storage" if mean_phi >= 0.15 else "Tight / Low Porosity"),
        "Key Comparison Notes & Concluding Remarks": phi_notes
    })

    # Optional Deep Resistivity RT row if RT was logged
    if mean_rt is not None:
        rt_prox = max(0.0, min(100.0, (np.log10(max(1.0, mean_rt)) / np.log10(85.0)) * 100.0))
        if mean_rt >= 30.0:
            rt_notes = f"High resistivity ({mean_rt:.1f} Ohm.m meets >=25 Ohm.m pay ideal). Non-conductive hydrocarbon column strongly confirmed; sharp resistivity contrast against underlying saline aquifer."
            rt_verdict = "Hydrocarbon Pay Zone"
        elif mean_rt >= 10.0:
            rt_notes = f"Moderate resistivity ({mean_rt:.1f} Ohm.m). Indicates oil-water transition zone, low-resistivity shaly sand pay, or fresh formation water; verify with fluid samples."
            rt_verdict = "Transition / Low-Resistivity"
        else:
            rt_notes = f"Low resistivity ({mean_rt:.1f} Ohm.m fails 15 Ohm.m pay cutoff). Pores saturated with highly conductive saline brine or conductive clay network; high water cut risk upon completion."
            rt_verdict = "Water-Bearing / Wet"

        rows.append({
            "Reservoir Parameter": "Deep Formation Resistivity (RT)",
            "Given Well Data": f"{mean_rt:.1f} Ohm.m",
            "Ideal Benchmark Standard": ">= 40.0 - 85.0 Ohm.m (Pay Standard)",
            "Engineering Tolerance Cutoff": ">= 15.0 Ohm.m (Pay Cutoff)",
            "Match Proximity": f"{rt_prox:.1f}%",
            "Evaluation Verdict": rt_verdict,
            "Key Comparison Notes & Concluding Remarks": rt_notes
        })

    # Hydrocarbon Saturation Row
    rows.append({
        "Reservoir Parameter": "Hydrocarbon Saturation (1 - Sw)",
        "Given Well Data": f"{sh_sat:.1%}",
        "Ideal Benchmark Standard": ">= 80.0% (Clean Hydrocarbon)",
        "Engineering Tolerance Cutoff": ">= 60.0% (Pay Cutoff)",
        "Match Proximity": f"{analysis.get('hydrocarbon_saturation_index', 0):.1f}%",
        "Evaluation Verdict": "Commercial Hydrocarbon Pay" if sh_sat >= 0.65 else ("Transitional Saturation" if sh_sat >= 0.50 else "High Water Cut / Wet"),
        "Key Comparison Notes & Concluding Remarks": sw_notes
    })

    return pd.DataFrame(rows)
