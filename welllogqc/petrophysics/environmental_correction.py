"""
welllogqc/petrophysics/environmental_correction.py
==================================================
Automated Borehole Environmental Correction Engine.

When caliper flags borehole enlargement, breakout, or washout (CALI > BS),
this module automatically applies physics-based borehole fluid, mudcake,
and pad-standoff corrections to:
1. Bulk Density (RHOB) -> neutralizes artificial density depression caused by mud standoff.
2. True Resistivity (RT) -> restores apparent resistivity depressed by conductive mud shunting.
3. Petrophysical curves (PHID, PHIE, SW, NTG) -> re-evaluated on corrected curves.

Based on industry-standard literature:
- Schlumberger: Log Interpretation Charts (FDC/LDT Borehole Correction & Tornado Curves)
- Ellis, D.V. & Singer, J.M.: Well Logging for Earth Scientists (2007)
- SPWLA Environmental Correction & Quality Standards
- Gardner, G.H.F. et al.: Formation Velocity and Density (Geophysics, 1974)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def compute_borehole_standoff(
    cali: pd.Series | np.ndarray,
    bs: float | pd.Series | np.ndarray = 8.5
) -> pd.Series:
    """
    Compute caliper enlargement over bit size (Delta-CALI in inches).
    
    Positive values indicate borehole breakout / washout.
    Negative values indicate mudcake buildup / tight hole.
    """
    cali_s = cali if isinstance(cali, pd.Series) else pd.Series(cali)
    if isinstance(bs, pd.Series):
        if len(bs) == len(cali_s):
            delta_cali = (cali_s - bs.values).clip(lower=-2.0, upper=15.0)
        else:
            delta_cali = (cali_s - bs).clip(lower=-2.0, upper=15.0)
    elif isinstance(bs, np.ndarray):
        delta_cali = (cali_s - bs).clip(lower=-2.0, upper=15.0)
    else:
        delta_cali = (cali_s - float(bs)).clip(lower=-2.0, upper=15.0)
    return delta_cali


def correct_density_washout(
    rhob: pd.Series,
    cali: pd.Series,
    bs: float | pd.Series = 8.5,
    dt: Optional[pd.Series] = None,
    rho_matrix: float = 2.65,
    rho_mud: float = 1.15
) -> Tuple[pd.Series, pd.Series]:
    """
    Automatically correct Bulk Density (RHOB) for borehole washout and pad standoff.
    
    Physics:
    In washouts, pad contact is compromised. The sensor measures a mixture of
    formation rock (rho_form) and drilling mud (rho_mud), artificially depressing RHOB.
    
    Correction Scheme:
    1. Moderate Washout (0.2 < Delta-CALI <= 1.5 in):
       Applies Schlumberger spine-and-ribs empirical standoff compensation:
       Delta_rho = 0.082 * (Delta-CALI) * (rho_matrix - rho_mud) / 1.50
    2. Severe Washout (Delta-CALI > 1.5 in):
       Pad standoff exceeds gamma-ray depth of investigation.
       If sonic (DT) is available: Gardner/Wyllie rock physics reconstruction is blended
       to restore the true formation density.
       Otherwise: progressive standoff compensation with asymptotic cap at matrix density.
       
    Returns:
        (rhob_corrected, delta_rho_applied)
    """
    rhob_clean = rhob.copy()
    delta_cali = compute_borehole_standoff(cali, bs)
    
    delta_rho = pd.Series(0.0, index=rhob.index)
    
    # Washout mask
    washout_mask = delta_cali > 0.20
    
    if washout_mask.any():
        # 1. Empirical standoff correction
        density_contrast = max(0.5, rho_matrix - rho_mud)
        dcal_washout = delta_cali[washout_mask]
        
        # Scaling factor calibrated against Schlumberger LDT charts
        standoff_corr = 0.082 * dcal_washout * (density_contrast / 1.50)
        delta_rho.loc[washout_mask] = standoff_corr
        
        # 2. Sonic Gardner fallback for severe washouts (> 1.5 in)
        severe_mask = washout_mask & (delta_cali > 1.50)
        if severe_mask.any() and dt is not None and not dt.dropna().empty:
            dt_severe = dt.loc[severe_mask].clip(lower=40.0, upper=200.0)
            # Wyllie acoustic porosity -> synthetic density
            phi_sonic = ((dt_severe - 55.5) / (189.0 - 55.5)).clip(lower=0.0, upper=0.45)
            rho_reconstructed = (1.0 - phi_sonic) * rho_matrix + phi_sonic * rho_mud
            
            # Blend: max between raw + standoff and reconstructed sonic density
            for idx in dt_severe.index:
                cand_corr = rhob_clean.loc[idx] + standoff_corr.loc[idx]
                recon = rho_reconstructed.loc[idx]
                best_rho = max(cand_corr, recon)
                delta_rho.loc[idx] = max(0.0, best_rho - rhob_clean.loc[idx])
            
    # Apply additive correction, capped at physical bounds [1.80, rho_matrix]
    rhob_corrected = (rhob_clean + delta_rho).clip(lower=1.80, upper=rho_matrix)
    delta_rho_actual = rhob_corrected - rhob_clean
    
    return rhob_corrected, delta_rho_actual


def correct_resistivity_washout(
    rt: pd.Series,
    cali: pd.Series,
    bs: float | pd.Series = 8.5,
    rm: float = 0.80
) -> Tuple[pd.Series, pd.Series]:
    """
    Automatically correct Deep Formation Resistivity (RT) for borehole enlargement.
    
    Physics:
    In washouts, conductive drilling mud (Rm << Rt) shunts electrical current,
    artificially depressing apparent deep resistivity (cave / borehole effect).
    
    Correction Scheme:
    Based on Schlumberger induction and dual laterolog borehole departure curves:
    Rt_corr = Rt_raw * [1 + alpha * (Delta-CALI / BS)^1.2 * max(0, log10(Rt_raw / Rm))]
    
    Returns:
        (rt_corrected, rt_multiplier_applied)
    """
    rt_clean = rt.clip(lower=0.05, upper=10000.0).copy()
    delta_cali = compute_borehole_standoff(cali, bs)
    
    washout_mask = delta_cali > 0.25
    multiplier = pd.Series(1.0, index=rt.index)
    
    if washout_mask.any():
        dcal = delta_cali[washout_mask]
        rt_sub = rt_clean[washout_mask]
        
        # Ensure bs_sub is strictly aligned with the filtered washout points
        if isinstance(bs, pd.Series):
            if bs.index.equals(rt.index):
                bs_sub = bs.loc[washout_mask]
            else:
                bs_sub = pd.Series(bs.values[washout_mask.values], index=dcal.index)
        elif isinstance(bs, np.ndarray):
            bs_sub = pd.Series(bs[washout_mask.values], index=dcal.index)
        else:
            bs_sub = float(bs)
            
        # Dimensionless ratios
        geom_ratio = (dcal / bs_sub).clip(lower=0.0, upper=2.0)
        contrast_ratio = np.maximum(0.0, np.log10(np.maximum(1.0, rt_sub / rm)))
        
        # Geometrical factor scaling (calibrated from induction departure charts)
        alpha = 0.18
        mult_sub = 1.0 + alpha * (geom_ratio ** 1.2) * contrast_ratio
        
        # Cap multiplier to physical maximum (+85% adjustment max)
        multiplier.loc[washout_mask] = mult_sub.clip(lower=1.0, upper=1.85)
        
    rt_corrected = rt_clean * multiplier
    return rt_corrected, multiplier


def apply_automated_environmental_corrections(
    df: pd.DataFrame,
    bs_default: float = 8.5,
    rho_matrix: float = 2.65,
    rho_fluid: float = 1.00,
    rho_mud: float = 1.15,
    rw: float = 0.05,
    rm: float = 0.80,
    gr_clean: float = 15.0,
    gr_shale: float = 120.0
) -> Dict[str, Any]:
    """
    Executes the full automated environmental correction pipeline.
    
    1. Detects caliper washouts against bit size.
    2. Corrects Bulk Density (RHOB) -> neutralizes pad standoff.
    3. Corrects Deep Resistivity (RT) -> neutralizes conductive mud shunting.
    4. Recomputes Density Porosity (PHID), Effective Porosity (PHIE), and Archie Sw.
    5. Returns an environmental correction package containing corrected curves,
       applied offsets, and diagnostic impact metrics.
    """
    cali = df["CALI"] if "CALI" in df.columns else None
    bs = df["BS"] if "BS" in df.columns else bs_default
    dt = df["DT"] if "DT" in df.columns else None
    delta_cali = compute_borehole_standoff(cali, bs) if cali is not None else None
    
    corrections_applied = False
    corr_df = df.copy()
    
    delta_rho_series = pd.Series(0.0, index=df.index)
    rt_multiplier_series = pd.Series(1.0, index=df.index)
    
    # 1. Density correction
    if "RHOB" in df.columns and cali is not None:
        rhob_corr, delta_rho = correct_density_washout(
            df["RHOB"], cali=cali, bs=bs, dt=dt,
            rho_matrix=rho_matrix, rho_mud=rho_mud
        )
        corr_df["RHOB_CORR"] = rhob_corr
        corr_df["DELTA_RHOB_APPLIED"] = delta_rho
        delta_rho_series = delta_rho
        if (delta_rho > 0.01).any():
            corrections_applied = True
            
    # 2. Resistivity correction
    if "RT" in df.columns and cali is not None:
        rt_corr, mult_rt = correct_resistivity_washout(
            df["RT"], cali=cali, bs=bs, rm=rm
        )
        corr_df["RT_CORR"] = rt_corr
        corr_df["RT_MULTIPLIER_APPLIED"] = mult_rt
        rt_multiplier_series = mult_rt
        if (mult_rt > 1.02).any():
            corrections_applied = True
            
    # 3. Recalculate Petrophysical Curves with Environmental Corrections
    petro_raw = {}
    petro_corr = {}
    
    # Shale volume
    if "GR" in df.columns:
        vsh_raw = ((df["GR"] - gr_clean) / (gr_shale - gr_clean)).clip(lower=0.0, upper=1.0)
        petro_raw["VSH"] = vsh_raw
        petro_corr["VSH"] = vsh_raw
        corr_df["VSH"] = vsh_raw
        
    # Porosity
    if "RHOB" in df.columns:
        phid_raw = ((rho_matrix - df["RHOB"]) / (rho_matrix - rho_fluid)).clip(lower=0.0, upper=0.45)
        petro_raw["PHID"] = phid_raw
        
        rhob_active = corr_df.get("RHOB_CORR", df["RHOB"])
        phid_corr = ((rho_matrix - rhob_active) / (rho_matrix - rho_fluid)).clip(lower=0.0, upper=0.45)
        petro_corr["PHID"] = phid_corr
        corr_df["PHID_CORR"] = phid_corr
        
        if "VSH" in petro_corr:
            petro_raw["PHIE"] = (phid_raw * (1.0 - petro_raw["VSH"])).clip(lower=0.0)
            petro_corr["PHIE"] = (phid_corr * (1.0 - petro_corr["VSH"])).clip(lower=0.0)
            corr_df["PHIE_CORR"] = petro_corr["PHIE"]
            
    elif "DT" in df.columns:
        phid_dt = ((df["DT"] - 55.5) / (189.0 - 55.5)).clip(lower=0.0, upper=0.45)
        petro_raw["PHID"] = phid_dt
        petro_corr["PHID"] = phid_dt
        corr_df["PHID_CORR"] = phid_dt
        if "VSH" in petro_corr:
            petro_corr["PHIE"] = (phid_dt * (1.0 - petro_corr["VSH"])).clip(lower=0.0)
            corr_df["PHIE_CORR"] = petro_corr["PHIE"]

    # Water Saturation Archie
    if "RT" in df.columns and "PHID" in petro_raw:
        phi_raw = petro_raw["PHID"].replace(0.0, 0.001)
        sw_raw = np.sqrt(np.maximum(0.0, (1.0 * rw) / ((phi_raw ** 2.0) * df["RT"].clip(lower=0.1)))).clip(lower=0.0, upper=1.0)
        petro_raw["SW"] = pd.Series(sw_raw, index=df.index)
        
        phi_c = petro_corr["PHID"].replace(0.0, 0.001)
        rt_c = corr_df.get("RT_CORR", df["RT"]).clip(lower=0.1)
        sw_corr = np.sqrt(np.maximum(0.0, (1.0 * rw) / ((phi_c ** 2.0) * rt_c))).clip(lower=0.0, upper=1.0)
        petro_corr["SW"] = pd.Series(sw_corr, index=df.index)
        corr_df["SW_CORR"] = petro_corr["SW"]
        corr_df["SH_CORR"] = (1.0 - petro_corr["SW"]).clip(lower=0.0, upper=1.0)

    # 4. Summary Impact Metrics
    washout_levels = int((delta_cali > 0.50).sum()) if cali is not None else 0
    mean_density_corr = float(delta_rho_series[delta_rho_series > 0.001].mean()) if (delta_rho_series > 0.001).any() else 0.0
    mean_rt_boost = float((rt_multiplier_series[rt_multiplier_series > 1.01] - 1.0).mean() * 100.0) if (rt_multiplier_series > 1.01).any() else 0.0
    
    return {
        "corrections_applied": corrections_applied,
        "corrected_df": corr_df,
        "petro_raw": petro_raw,
        "petro_corr": petro_corr,
        "washout_levels_count": washout_levels,
        "mean_density_correction_g_cm3": round(mean_density_corr, 3),
        "mean_resistivity_boost_pct": round(mean_rt_boost, 1),
        "delta_rho_series": delta_rho_series,
        "rt_multiplier_series": rt_multiplier_series,
    }
