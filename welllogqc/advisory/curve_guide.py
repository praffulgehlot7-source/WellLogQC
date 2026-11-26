"""
welllogqc/advisory/curve_guide.py
=================================
Authoritative Petrophysical Reference Guide, Ideal Log Standards,
and Dynamic Hover Point Evaluator.

Compiled from industry-standard petrophysical literature:
- Schlumberger: Log Interpretation Principles & Applications (1989/2009)
- Asquith, G. & Krygowski, D.: Basic Well Log Analysis, AAPG Methods in Exploration 16 (2004)
- SPWLA (Society of Petrophysicists and Well Log Analysts) Log Quality Guidelines
- Glover, P.W.J.: Petrophysics MSc Course Notes, University of Aberdeen
- Ellis, D.V. & Singer, J.M.: Well Logging for Earth Scientists, Springer (2007)

Strictly zero emojis. All descriptions written in user-friendly plain English.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Authoritative Curve Knowledge Base
# ---------------------------------------------------------------------------
CURVE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "GR": {
        "name": "Gamma Ray",
        "unit": "gAPI",
        "min_value": 0.0,
        "max_value": 300.0,
        "category": "Lithology & Clay Fraction",
        "physics": "Measures total natural radioactivity from Potassium (40K), Uranium (238U), and Thorium (232Th) using scintillation crystal detectors.",
        "ideal_value": "15.0 - 45.0 gAPI (Clean Quartz Sandstone)",
        "ideal_range": (15.0, 45.0),
        "cutoff_value": "<= 60.0 gAPI (Shale Cutoff for Reservoir Pay)",
        "shale_baseline": ">= 90.0 - 150.0+ gAPI (Pure Shale / Marine Mudstone)",
        "significance": "Primary tool for distinguishing reservoir rocks (clean sandstone, limestone) from non-reservoir cap rocks (shale). Essential for calculating shale volume (Vsh).",
        "high_meaning": "High GR (>75 gAPI) indicates clay-rich shale, silt, or radioactive minerals (e.g., feldspars, uranium-rich organic shales). Implies zero to low effective permeability.",
        "low_meaning": "Low GR (<45 gAPI) indicates clean quartz sandstone or clean carbonate. Represents prime reservoir matrix with minimal pore-clogging clay.",
        "common_pitfalls": "Radioactive sandstone (arkosic sands with potassium feldspar or heavy minerals) can exhibit high GR despite excellent reservoir porosity. Uranium-rich fractures may also cause misleading spikes.",
        "reference": "Schlumberger Log Interpretation Principles (1989), Asquith & Krygowski (AAPG, 2004)."
    },
    "CALI": {
        "name": "Borehole Caliper",
        "unit": "inches (in)",
        "min_value": 4.0,
        "max_value": 24.0,
        "category": "Borehole Quality & Sensor Integrity",
        "physics": "Measures the physical diameter of the borehole using spring-loaded mechanical arms contacting the borehole wall.",
        "ideal_value": "Matches Bit Size (+/- 0.20 in on-gauge hole)",
        "ideal_range": (0.0, 0.25),  # deviation from BS
        "cutoff_value": "Max Breakout: Bit Size + 1.50 in",
        "shale_baseline": "CALI >> Bit Size (+2.0 to +6.0 in in washouts)",
        "significance": "Crucial QC indicator for the entire logging suite. Pad contact tools (RHOB, NPHI) require a smooth, on-gauge borehole to read accurate formation properties.",
        "high_meaning": "Caliper significantly larger than bit size indicates borehole washout, breakout, or sloughing shales caused by mechanical collapse or water-based mud erosion.",
        "low_meaning": "Caliper smaller than bit size indicates mudcake buildup across permeable formations (good sign of permeability) or creeping/swelling formations (risk of stuck tool).",
        "common_pitfalls": "Pad-mounted density and neutron tools lose wall contact in washouts, measuring mud column instead of rock, causing false 'gas effect' or artificially high porosity.",
        "reference": "SPWLA Log Quality Control Guidelines; Schlumberger Wireline Reference."
    },
    "BS": {
        "name": "Nominal Bit Size",
        "unit": "inches (in)",
        "min_value": 4.0,
        "max_value": 24.0,
        "category": "Drilling Reference Geometry",
        "physics": "Nominal outer diameter of the drill bit that bored the hole interval.",
        "ideal_value": "Equal to Caliper (CALI == BS)",
        "ideal_range": (6.0, 12.25),
        "cutoff_value": "Baseline Reference",
        "shale_baseline": "Constant gauge reference",
        "significance": "Serves as the baseline geometric reference against which borehole enlargement (washout) and constriction (mudcake) are quantified.",
        "high_meaning": "Larger hole size (e.g., 12.25 in) requires greater tool pad contact and volume corrections.",
        "low_meaning": "Slim hole geometry (e.g., 6.0 in) reduces mud volume but increases tool sticking risks.",
        "common_pitfalls": "Incorrect bit size recorded in LAS header distorts washout calculations. Always verify against drilling daily report.",
        "reference": "API Recommended Practice 40; Schlumberger Log Interpretation Charts."
    },
    "RHOB": {
        "name": "Bulk Density",
        "unit": "g/cm3",
        "min_value": 1.10,
        "max_value": 3.20,
        "category": "Matrix Density & Total Porosity",
        "physics": "Emits gamma rays from a Cesium-137 source and measures back-scattered gamma flux via Compton scattering, which is directly proportional to electron density.",
        "ideal_value": "2.15 - 2.30 g/cm3 (for 21% - 30% porous sandstone)",
        "ideal_range": (2.10, 2.35),
        "cutoff_value": "<= 2.40 g/cm3 (for >= 15% Porosity in Sandstone)",
        "shale_baseline": "2.45 - 2.68 g/cm3 (Dense, compacted shale)",
        "significance": "Fundamental quantitative measurement for determining total reservoir porosity and identifying gas caps via density-neutron crossplots.",
        "high_meaning": "High RHOB (>2.50 g/cm3) indicates tight, cemented rock matrix (limestone/dolomite), dense shales, or heavy minerals (pyrite, siderite).",
        "low_meaning": "Low RHOB (<2.25 g/cm3) indicates high porosity, light hydrocarbons, or natural gas (gas has very low density ~0.2 g/cm3).",
        "common_pitfalls": "Borehole washouts cause pad standoff: sensor reads light drilling mud (1.0 - 1.2 g/cm3) instead of rock, yielding artificially low density and inflated porosity.",
        "reference": "Ellis & Singer, Well Logging for Earth Scientists (2007); Schlumberger (1989)."
    },
    "NPHI": {
        "name": "Neutron Porosity",
        "unit": "v/v or dec (limestone units)",
        "min_value": -0.05,
        "max_value": 0.60,
        "category": "Hydrogen Index & Gas Detection",
        "physics": "Bombards formation with fast neutrons from an Americium-Beryllium source; collisions with hydrogen nuclei (water, oil, gas) slow neutrons down to thermal energies.",
        "ideal_value": "0.18 - 0.28 v/v (Tracks Density Porosity in Liquid Sand)",
        "ideal_range": (0.15, 0.30),
        "cutoff_value": "Cross-over with Density Porosity",
        "shale_baseline": "0.35 - 0.55+ v/v (Bound water in clay lattices)",
        "significance": "Primary hydrogen concentration sensor. In liquid-filled sandstones, NPHI matches true porosity; combined with RHOB, it identifies gas zones and clay volume.",
        "high_meaning": "High NPHI (>0.35) in non-gas zones indicates clay minerals: clay crystal lattices hold structural bound water (OH-), tricking the sensor into seeing high porosity.",
        "low_meaning": "Low NPHI (<0.15) indicates tight matrix or gas zones: gas has low hydrogen index compared to water or oil, causing apparent NPHI to drop sharply.",
        "common_pitfalls": "Gas crossover: When NPHI drops below RHOB-derived porosity, gas is present. However, borehole washouts also cause apparent crossover due to mud dilution.",
        "reference": "Schlumberger Log Interpretation Charts; Asquith & Krygowski (2004)."
    },
    "RT": {
        "name": "Deep Resistivity",
        "unit": "Ohm.m",
        "min_value": 0.1,
        "max_value": 5000.0,
        "category": "Fluid Saturation & Pay Identification",
        "physics": "Measures true formation electrical resistivity (Rt) deep beyond the invaded mud filtrate zone using electromagnetic induction coils or deep galvanic electrode arrays.",
        "ideal_value": "25.0 - 150.0+ Ohm.m (Hydrocarbon-Filled Reservoir)",
        "ideal_range": (20.0, 500.0),
        "cutoff_value": ">= 10.0 - 20.0 Ohm.m (Pay Cutoff vs. Water Wet)",
        "shale_baseline": "0.8 - 4.0 Ohm.m (Conductive Saline Clay System)",
        "significance": "The single most critical log for hydrocarbon detection. Non-conductive oil and gas resist electric current, whereas saline formation water conducts electricity easily.",
        "high_meaning": "High RT (>20 Ohm.m) in a porous sand indicates hydrocarbons (oil/gas) filling pore space. In tight rocks, high RT can also indicate non-porous dense matrix.",
        "low_meaning": "Low RT (<5 Ohm.m) indicates 100% saline water saturation (aquifer) or conductive clay-rich shales.",
        "common_pitfalls": "Low-resistivity pay: Thinly laminated sand-shale beds or conductive minerals (pyrite) can depress RT in commercially productive hydrocarbon zones.",
        "reference": "Archie, G.E. (1942), AIME Trans; Schlumberger Log Interpretation Principles (1989)."
    },
    "DT": {
        "name": "Acoustic Sonic Transit Time",
        "unit": "us/ft (microseconds per foot)",
        "min_value": 40.0,
        "max_value": 240.0,
        "category": "Acoustic Porosity & Mechanical Geomechanics",
        "physics": "Measures interval transit time (slowness, Delta-t) of acoustic compressional sound waves traveling through the rock matrix and pore fluid.",
        "ideal_value": "75.0 - 95.0 us/ft (18% - 30% Porous Sandstone)",
        "ideal_range": (70.0, 100.0),
        "cutoff_value": ">= 68.0 us/ft (>= 10% Sonic Porosity)",
        "shale_baseline": "95.0 - 140.0+ us/ft (Soft, uncompacted shale)",
        "significance": "Provides an independent porosity calculation via Wyllie's time-average equation that is less sensitive to borehole washouts than density or neutron tools.",
        "high_meaning": "High DT (>95 us/ft) indicates high porosity, undercompacted sediments, soft shales, or fracture zones (sound waves travel slower through fluid and gas).",
        "low_meaning": "Low DT (<60 us/ft) indicates tight, cemented rocks (pure quartz matrix is 55.5 us/ft; limestone is 47.5 us/ft; dolomite is 43.5 us/ft).",
        "common_pitfalls": "Cycle skipping: In gas-bearing, fractured, or washed-out formations, acoustic wave attenuation can cause the detector to miss the first wave peak, creating false spikes.",
        "reference": "Wyllie, M.R.J. et al. (1956), Geophysics; Glover, Petrophysics MSc Notes."
    },
    "PEF": {
        "name": "Photoelectric Absorption Factor",
        "unit": "b/e (barns per electron)",
        "min_value": 0.5,
        "max_value": 15.0,
        "category": "Matrix Lithology Identification",
        "physics": "Measures low-energy gamma ray absorption, which is directly related to the average atomic number (Z) of formation rock atoms.",
        "ideal_value": "1.81 b/e (Sandstone Matrix) | 5.08 b/e (Limestone) | 3.14 b/e (Dolomite)",
        "ideal_range": (1.6, 2.2),
        "cutoff_value": "Lithology Indicator: Sandstone ~ 1.8 b/e",
        "shale_baseline": "2.5 - 4.5 b/e (Clay mineral mixtures)",
        "significance": "Directly identifies mineral matrix lithology regardless of porosity. Crucial for complex mixed sand-carbonate-shale sequences.",
        "high_meaning": "High PEF (>4.5 b/e) indicates limestone or heavy barite drilling mud additive. Very high (>10 b/e) indicates barite mud contamination.",
        "low_meaning": "Low PEF (~1.8 b/e) confirms clean quartz sandstone lithology.",
        "common_pitfalls": "Barite in drilling mud has an extremely high PEF (~267 b/e), which obliterates formation PEF readings if mud invasion is deep.",
        "reference": "Schlumberger Litho-Density Tool Technical Manual; Asquith & Krygowski (2004)."
    },
    "SP": {
        "name": "Spontaneous Potential",
        "unit": "millivolts (mV)",
        "min_value": -120.0,
        "max_value": 60.0,
        "category": "Permeability & Formation Water Salinity",
        "physics": "Measures direct-current electrical potential generated by electrochemical interactions between drilling mud filtrate and saline formation water.",
        "ideal_value": "Negative deflection (-30 to -80 mV) in permeable sands with saline water",
        "ideal_range": (-80.0, -20.0),
        "cutoff_value": "Deflection from shale baseline",
        "shale_baseline": "Shale baseline (straight reference line near 0 mV)",
        "significance": "Identifies permeable beds and estimates formation water resistivity (Rw).",
        "high_meaning": "Zero deflection indicates impermeable shales or fresh water where mud salinity equals formation water salinity.",
        "low_meaning": "Strong negative deflection indicates permeable sand filled with formation water more saline than mud filtrate.",
        "common_pitfalls": "SP is disabled in non-conductive oil-based muds (OBM). Telluric currents and magnetised casing also induce severe baseline drift.",
        "reference": "Doll, H.G. (1948), The SP Log: Theoretical Analysis; AAPG Wiki."
    },
    "MSFL": {
        "name": "Micro-Spherically Focused Log (Rxo)",
        "unit": "Ohm.m",
        "min_value": 0.1,
        "max_value": 1000.0,
        "category": "Flushed Zone Resistivity & Movable Oil",
        "physics": "Measures resistivity of the shallow flushed zone (Rxo) immediately adjacent to the borehole wall using pad-mounted electrodes with spherical focusing.",
        "ideal_value": "Variable (Rxo > Rt indicates filtrate invasion in pay)",
        "ideal_range": (1.0, 50.0),
        "cutoff_value": "Permeability & Movable Oil Indicator (Rxo / Rt ratio)",
        "shale_baseline": "Rxo ≈ Rt in impermeable shales",
        "significance": "Quantifies mud filtrate invasion and calculates movable hydrocarbon saturation: (Sxo - Sw).",
        "high_meaning": "High Rxo indicates resistive mud filtrate invasion in permeable formation.",
        "low_meaning": "Low Rxo indicates conductive saline mud invasion or shale.",
        "common_pitfalls": "Mudcake thickness > 0.5 in distorts Rxo pad electrode readings.",
        "reference": "Schlumberger Wireline Essentials; Asquith & Krygowski (2004)."
    },
    "VSH": {
        "name": "Volume of Shale (Vsh)",
        "unit": "v/v (fraction or %)",
        "min_value": 0.0,
        "max_value": 1.0,
        "category": "Petrophysical Reservoir Cleanliness",
        "physics": "Calculated from Gamma Ray baseline indexing: Vsh = (GR - GR_clean) / (GR_shale - GR_clean).",
        "ideal_value": "0.0 - 0.15 (0% - 15% clean quartz sand)",
        "ideal_range": (0.0, 0.15),
        "cutoff_value": "<= 0.35 (35% Net Pay Cutoff)",
        "shale_baseline": ">= 0.70 (Non-reservoir shale unit)",
        "significance": "Dictates effective porosity and permeability. Clay minerals clog pore throats and trap immobile bound water.",
        "high_meaning": "High Vsh (>0.35) reduces pore throat radii, destroying fluid transmissibility. Excluded from commercial net pay.",
        "low_meaning": "Low Vsh (<0.15) provides open, interconnected pore networks ideal for hydrocarbon drainage.",
        "common_pitfalls": "Linear Vsh is conservative. For consolidated older rocks, non-linear Steiber or Larionov corrections may yield higher net pay.",
        "reference": "Steiber (1970); Larionov (1969); Asquith & Krygowski (2004)."
    },
    "PHID": {
        "name": "Density Porosity (phiD)",
        "unit": "v/v (fraction or %)",
        "min_value": 0.0,
        "max_value": 0.45,
        "category": "Total Matrix Void Volume",
        "physics": "Calculated from bulk density: phiD = (rho_matrix - RHOB) / (rho_matrix - rho_fluid).",
        "ideal_value": "0.20 - 0.28 (20% - 28% commercial storage)",
        "ideal_range": (0.20, 0.32),
        "cutoff_value": ">= 0.10 (10% Commercial Pay Cutoff)",
        "shale_baseline": "< 0.08 (Tight or non-reservoir shale)",
        "significance": "Quantifies the physical void space within the rock matrix available to store hydrocarbons.",
        "high_meaning": "High porosity (>20%) yields massive hydrocarbons in place (STOIIP) and high recovery factors.",
        "low_meaning": "Low porosity (<10%) limits commercial flow rates unless stimulated by hydraulic fracturing.",
        "common_pitfalls": "Must use accurate matrix density (2.65 for sand, 2.71 for limestone). Washouts artificially inflate phiD.",
        "reference": "Schlumberger Log Interpretation Principles (1989)."
    },
    "PHIE": {
        "name": "Effective Porosity (phiE)",
        "unit": "v/v (fraction or %)",
        "min_value": 0.0,
        "max_value": 0.40,
        "category": "Interconnected Effective Pore Volume",
        "physics": "Total porosity discounted for clay-bound water: phiE = phiD * (1 - Vsh).",
        "ideal_value": "0.18 - 0.26 (18% - 26% free fluid pore volume)",
        "ideal_range": (0.16, 0.30),
        "cutoff_value": ">= 0.08 (8% Commercial Flow Cutoff)",
        "shale_baseline": "< 0.04 (Negligible interconnected pore space)",
        "significance": "True fluid storage volume capable of contributing to commercial production.",
        "high_meaning": "High phiE guarantees robust commercial flow rates and high recovery factor.",
        "low_meaning": "Low phiE (<8%) indicates tight pore throats or heavy clay clogging.",
        "common_pitfalls": "Ignoring clay correction results in overestimation of recoverable hydrocarbon reserves.",
        "reference": "Asquith & Krygowski (2004); Worthington, P.F. (1985)."
    },
    "SW": {
        "name": "Water Saturation (Archie Sw)",
        "unit": "v/v (fraction or %)",
        "min_value": 0.0,
        "max_value": 1.0,
        "category": "Fluid Saturation & Net Pay",
        "physics": "Calculated via Archie's Law: Sw = ((a / phi^m) * (Rw / Rt))^(1/n).",
        "ideal_value": "0.15 - 0.30 (15% - 30% irreducible water; 70% - 85% Hydrocarbons)",
        "ideal_range": (0.10, 0.35),
        "cutoff_value": "<= 0.50 (50% Water Cutoff for Commercial Pay)",
        "shale_baseline": "1.00 (100% Water wet or shale bound water)",
        "significance": "Determines whether a reservoir will produce oil, gas, or formation water upon perforation.",
        "high_meaning": "High Sw (>50%) indicates high risk of water production upon completion.",
        "low_meaning": "Low Sw (<30%) confirms prime hydrocarbon pay with minimal water cut.",
        "common_pitfalls": "Archie equation assumes clean sand. In shaly sands, clay conductivity depresses Rt, causing Archie to overestimate Sw (use Waxman-Smits or Simandoux).",
        "reference": "Archie, G.E. (1942); Worthington, P.F. (1985), The Evolution of Shaly-Sand Concepts."
    },
    "SH": {
        "name": "Hydrocarbon Saturation (Sh)",
        "unit": "v/v (fraction or %)",
        "min_value": 0.0,
        "max_value": 1.0,
        "category": "Hydrocarbon In-Place Fraction",
        "physics": "Complement of water saturation: Sh = 1 - Sw.",
        "ideal_value": "0.70 - 0.85 (70% - 85% Hydrocarbon Column)",
        "ideal_range": (0.65, 0.90),
        "cutoff_value": ">= 0.50 (50% Commercial Pay Cutoff)",
        "shale_baseline": "0.00 (Zero movable hydrocarbons)",
        "significance": "Direct fraction of pore space occupied by oil or natural gas.",
        "high_meaning": "High Sh confirms abundant movable hydrocarbons.",
        "low_meaning": "Low Sh indicates water-wet rock or transition zone.",
        "common_pitfalls": "Residual oil (Sor) may not be fully movable without secondary recovery.",
        "reference": "Schlumberger Log Interpretation Principles (1989)."
    },
    "NTG": {
        "name": "Net-to-Gross Ratio (NTG)",
        "unit": "dimensionless (ratio 0 - 1)",
        "min_value": 0.0,
        "max_value": 1.0,
        "category": "Formation Pay Quality Ratio",
        "physics": "Ratio of commercial productive net pay thickness to total gross logged interval.",
        "ideal_value": "0.75 - 0.95 (High-Quality Continuous Pay Sand)",
        "ideal_range": (0.65, 1.00),
        "cutoff_value": ">= 0.50 (Commercial Reservoir Development Cutoff)",
        "shale_baseline": "< 0.20 (Non-commercial / highly compartmentalized)",
        "significance": "Primary economic parameter driving field development, well spacing, and recovery forecasting.",
        "high_meaning": "High NTG indicates massive, unbroken commercial reservoir column.",
        "low_meaning": "Low NTG indicates highly laminated sand-shale sequences with limited drainage area.",
        "common_pitfalls": "Uncorrected washouts artificially inflate net pay if density porosity is not corrected.",
        "reference": "SPWLA Net Pay Guidelines (2008)."
    },
    "CGR": {
        "name": "Computed Gamma Ray (Th + K)",
        "unit": "gAPI",
        "min_value": 0.0,
        "max_value": 250.0,
        "category": "Spectral Lithology & Clay Typing",
        "physics": "Sum of Thorium (232Th) and Potassium (40K) radioactivity, subtracting Uranium (238U) to isolate pure clay matrix effects.",
        "ideal_value": "10.0 - 40.0 gAPI (Clean Quartz Sandstone)",
        "ideal_range": (10.0, 40.0),
        "cutoff_value": "<= 50.0 gAPI (Clay Cutoff for Net Pay)",
        "shale_baseline": ">= 80.0 - 140.0 gAPI (Pure Marine Shale)",
        "significance": "Crucial for identifying true clay volume in organic-rich shales or uranium-bearing sandstones where total GR is misleadingly high.",
        "high_meaning": "High CGR confirms clay mineral presence (illite, smectite), not organic matter.",
        "low_meaning": "Low CGR confirms true low-clay quartz sandstone even if total GR is high.",
        "common_pitfalls": "Heavy minerals (zircon, monazite) can slightly elevate thorium in clean sand.",
        "reference": "Hurst, A. (1990), Geological Society London; Schlumberger Spectrometry Guide."
    },
    "DRHO": {
        "name": "Density Correction (Delta-Rho)",
        "unit": "g/cm3",
        "min_value": -0.10,
        "max_value": 0.45,
        "category": "Borehole Mudcake & Stand-off Quality",
        "physics": "Difference between short-spaced and long-spaced density detectors, quantifying mudcake thickness and pad standoff compensation.",
        "ideal_value": "-0.02 to +0.03 g/cm3 (Smooth on-gauge borehole with minimal mudcake)",
        "ideal_range": (-0.02, 0.03),
        "cutoff_value": "|DRHO| <= 0.05 g/cm3 (High-Confidence Density); > 0.15 g/cm3 (Unreliable)",
        "shale_baseline": "Variable (+0.05 to +0.25 g/cm3 in rough borehole washouts)",
        "significance": "Primary confidence check for RHOB. Large DRHO indicates the density pad lost contact with formation rock.",
        "high_meaning": "DRHO > 0.10 g/cm3 indicates thick mudcake or severe borehole standoff. RHOB may be under-corrected.",
        "low_meaning": "DRHO near 0.0 confirms tight pad wall contact and high-accuracy density readings.",
        "common_pitfalls": "In heavy barite drilling muds, photoelectric absorption can cause DRHO to swing negative.",
        "reference": "Schlumberger Litho-Density Tool QC Manual; SPWLA Logging Standards."
    },
    "LLD": {
        "name": "Deep Laterolog (Rt)",
        "unit": "Ohm.m",
        "min_value": 0.1,
        "max_value": 5000.0,
        "category": "Deep Formation Resistivity (Rt)",
        "physics": "Measures uninvaded formation resistivity (Rt) in conductive/saline mud boreholes using guard-electrode focused electrical currents.",
        "ideal_value": "20.0 - 200.0+ Ohm.m (Hydrocarbon-Bearing Reservoir)",
        "ideal_range": (20.0, 500.0),
        "cutoff_value": ">= 10.0 - 20.0 Ohm.m (Hydrocarbon Pay Cutoff)",
        "shale_baseline": "0.8 - 4.0 Ohm.m (Conductive Saline Marine Shale)",
        "significance": "Standard deep resistivity measurement in high-salinity muds where induction tools lose dynamic range.",
        "high_meaning": "High LLD indicates hydrocarbons (oil/gas) filling matrix pore spaces, or tight dense carbonates.",
        "low_meaning": "Low LLD indicates 100% saline water saturation (aquifer) or clay-rich shale.",
        "common_pitfalls": "Thin resistive beds below tool vertical resolution are smoothed out, causing apparent resistivity to read lower than true Rt.",
        "reference": "Schlumberger Dual Laterolog Principles (1982); Asquith & Krygowski (2004)."
    },
    "LLS": {
        "name": "Shallow Laterolog (Rxo / Ri)",
        "unit": "Ohm.m",
        "min_value": 0.1,
        "max_value": 2000.0,
        "category": "Invaded Zone Resistivity (Ri)",
        "physics": "Measures formation resistivity at intermediate radial depth (15 - 30 in) within the mud filtrate invaded transition zone.",
        "ideal_value": "Profile separation from LLD indicates mud filtrate invasion into permeable rock",
        "ideal_range": (5.0, 100.0),
        "cutoff_value": "Invasion Profile: LLD > LLS indicates hydrocarbon flushing by saline mud",
        "shale_baseline": "LLS ≈ LLD (Impermeable shale, zero invasion)",
        "significance": "Combined with LLD to quantify the depth of mud filtrate invasion and determine true resistivity Rt.",
        "high_meaning": "High LLS indicates resistive oil-filtrate or resistive tight rock.",
        "low_meaning": "Low LLS indicates conductive saline mud filtrate invasion.",
        "common_pitfalls": "Severe borehole rugosity affects shallow arrays more than deep arrays.",
        "reference": "Ellis & Singer, Well Logging for Earth Scientists (2007)."
    },
    "DTS": {
        "name": "Shear Wave Sonic Slowness",
        "unit": "us/ft",
        "min_value": 70.0,
        "max_value": 400.0,
        "category": "Geomechanics & Shear Modulus",
        "physics": "Measures acoustic interval transit time for transverse shear sound waves generated by dipole acoustic transmitters.",
        "ideal_value": "110.0 - 150.0 us/ft (Porous sand reservoir)",
        "ideal_range": (100.0, 160.0),
        "cutoff_value": "Vp/Vs ratio calculation: DT/DTS ~ 1.55 - 1.75 in gas/oil sand",
        "shale_baseline": "180.0 - 320.0+ us/ft (Weak, ductile shale)",
        "significance": "Essential for calculating dynamic shear modulus (G), Poisson's ratio (nu), and rock mechanical brittleness for hydraulic fracturing.",
        "high_meaning": "High DTS indicates soft, poorly consolidated or uncompacted formation.",
        "low_meaning": "Low DTS indicates fast, stiff rock with high shear modulus and high fracture gradient.",
        "common_pitfalls": "Shear waves cannot propagate through borehole fluids; requires dipole flexural wave processing in slow formations.",
        "reference": "Mavko, G. et al., The Rock Physics Handbook (Cambridge, 2009); Glover (Aberdeen)."
    },
    "PERM": {
        "name": "Permeability (Hydraulic Conductivity)",
        "unit": "mD (millidarcies)",
        "min_value": 0.001,
        "max_value": 10000.0,
        "category": "Dynamic Fluid Flow Transmissibility",
        "physics": "Quantifies the rock's capacity to transmit fluids under pressure gradient; derived from Timur, Coates, or Tixier petrophysical transforms.",
        "ideal_value": "50.0 - 2000.0+ mD (High-Deliverability Commercial Pay)",
        "ideal_range": (20.0, 1500.0),
        "cutoff_value": ">= 1.0 mD (Conventional Gas Cutoff) | >= 5.0 mD (Light Oil Cutoff)",
        "shale_baseline": "< 0.01 mD (Impermeable Hydraulic Barrier / Caprock Seal)",
        "significance": "Determines open-flow production rate, required drawdown pressure, and ultimate economic recovery.",
        "high_meaning": "High permeability (>100 mD) allows massive commercial production rates without hydraulic stimulation.",
        "low_meaning": "Low permeability (<1 mD) requires artificial hydraulic stimulation or indicates non-reservoir tight matrix.",
        "common_pitfalls": "Log-derived permeability is an uncalibrated estimate; must be calibrated against core plug routine analysis.",
        "reference": "Timur, A. (1968), JPT; Coates, G.R. & Dumanoir, J.L. (1973), SPWLA."
    },
    "NMR_TCMR": {
        "name": "NMR Total Porosity (TCMR)",
        "unit": "v/v (fraction or %)",
        "min_value": 0.0,
        "max_value": 0.50,
        "category": "Nuclear Magnetic Resonance Total Pore Volume",
        "physics": "Measures hydrogen proton spin relaxation (T1 and T2 decay) in magnetic field gradient, directly yielding total fluid volume independent of mineral matrix.",
        "ideal_value": "0.20 - 0.32 v/v (Porous Reservoir)",
        "ideal_range": (0.18, 0.30),
        "cutoff_value": "Matches Density-Neutron Porosity in clean sands",
        "shale_baseline": "< 0.10 v/v (Bound fluid only in clay micro-porosity)",
        "significance": "Lithology-independent porosity tool that partitions pore space into clay-bound, capillary-bound, and free-producible fluid.",
        "high_meaning": "High NMR porosity confirms large volumetric fluid storage capacity.",
        "low_meaning": "Low NMR porosity indicates tight matrix or low fluid volume.",
        "common_pitfalls": "Under-polarization in high-viscosity heavy oils or fast tool logging speeds underestimates true porosity.",
        "reference": "Coates, G.R. et al., NMR Logging Principles and Applications (Halliburton, 1999)."
    },
    "TENS": {
        "name": "Cable Head Tension",
        "unit": "lbs (pounds-force)",
        "min_value": 500.0,
        "max_value": 12000.0,
        "category": "Wireline Cable Mechanics & Tool Safety",
        "physics": "Monitors pulling tension exerted on the armored wireline cable head by surface winch.",
        "ideal_value": "Smooth, continuous tension curve tracking tool string weight in mud (+/- 200 lbs)",
        "ideal_range": (1500.0, 4500.0),
        "cutoff_value": "< Safe Working Load (60% of cable breaking strength, typically 8000 lbs)",
        "shale_baseline": "Stable baseline during uphole logging pass",
        "significance": "Safety-critical diagnostic: detects differential sticking, dogleg friction, and downhole hang-ups.",
        "high_meaning": "Abrupt tension spikes indicate tool sticking, differential sticking against permeable beds, or keyseating in doglegs.",
        "low_meaning": "Sudden tension drop indicates slack line (tool hung up on downhole ledge during descent).",
        "common_pitfalls": "Ignoring tension cycling risks cable parting downhole and expensive fishing operations.",
        "reference": "API RP 67 (Recommended Practice for Wellsite Log Safety); Wireline Operational Manuals."
    }
}


# ---------------------------------------------------------------------------
# Dynamic Hover Point Evaluator
# ---------------------------------------------------------------------------
def evaluate_point_status(curve_name: str, value: float, aux: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """
    Evaluates a single measurement point for a given curve and returns:
      (status_label, plain_english_interpretation)
    
    Args:
        curve_name: Canonical curve name (e.g. 'GR', 'CALI', 'RT', 'RHOB', 'NPHI', 'SW', 'PHID')
        value: Numerical measurement at that depth
        aux: Optional auxiliary dictionary containing complementary data
             (e.g. {'BS': 8.5, 'PHID': 0.22, 'VSH': 0.12})
    """
    if pd.isna(value):
        return ("No Data", "Sensor reading absent or null at this depth.")
    
    aux = aux or {}
    c = str(curve_name).upper().strip()
    
    # ── 1. GAMMA RAY (GR) ──────────────────────────────────────────────────────
    if c == "GR":
        if value < 15.0:
            return (
                "Ultra-Clean Matrix (Quartz / Carbonate / Evaporite)",
                f"{value:.1f} gAPI is exceptionally low. Indicates pristine quartz sand, anhydrite, or pure limestone with zero clay content."
            )
        elif value <= 45.0:
            return (
                "Optimal (Clean Quartz Sandstone)",
                f"{value:.1f} gAPI is ideal (clean sand < 45 gAPI). Low radioactivity confirms clean quartz sand with minimal clay, prime permeability, and high reservoir quality."
            )
        elif value <= 65.0:
            return (
                "Good (Slightly Shaly Sand)",
                f"{value:.1f} gAPI is good. Minor silt or clay contamination; reservoir maintains commercial storage and fluid transmissibility."
            )
        elif value <= 85.0:
            return (
                "Moderate (Shaly Sandstone / Siltstone)",
                f"{value:.1f} gAPI indicates moderate clay fraction. Clay minerals reduce effective porosity and pore throat connectivity."
            )
        elif value <= 110.0:
            return (
                "High (Sandy Shale / Transition Unit)",
                f"{value:.1f} gAPI is high (ideal clean sand < 45 gAPI). Elevated radioactivity indicates a clay-dominated formation with marginal to poor flow potential."
            )
        else:  # e.g., 143 gAPI
            return (
                "Very High (Heavy Shale / Non-Reservoir Seal)",
                f"{value:.1f} gAPI is very high (ideal clean reservoir < 45 gAPI). Indicates clay-rich, tight shale or top-seal with zero effective permeability."
            )
            
    # ── 2. CALIPER (CALI) ─────────────────────────────────────────────────────
    elif c == "CALI":
        bs = float(aux.get("BS", 8.5))
        dev = value - bs
        if dev > 2.5:
            return (
                "Critical Washout (Severe Breakout)",
                f"Caliper ({value:.2f} in) exceeds bit size ({bs:.2f} in) by +{dev:.2f} in. Pad sensors (density/neutron) have lost wall contact; porosity readings are artificially inflated."
            )
        elif dev > 1.2:
            return (
                "Moderate Washout (Enlarged Hole)",
                f"Caliper ({value:.2f} in) exceeds bit size by +{dev:.2f} in. Borehole enlargement detected; density and neutron readings require environmental corrections."
            )
        elif abs(dev) <= 0.4:
            return (
                "Optimal (On-Gauge Borehole)",
                f"Caliper ({value:.2f} in) matches drill bit gauge ({bs:.2f} in) within {abs(dev):.2f} in. Pad sensors maintain firm formation contact for reliable evaluation."
            )
        elif dev < -0.4:
            return (
                "Tight Hole / Mudcake Buildup",
                f"Caliper ({value:.2f} in) is {abs(dev):.2f} in under bit size ({bs:.2f} in). Indicates permeable mudcake buildup or swelling formation; monitor for tool drag."
            )
        else:
            return (
                "Minor Caliper Deviation",
                f"Caliper ({value:.2f} in) is +{dev:.2f} in above bit size ({bs:.2f} in). Hole condition is stable."
            )

    # ── 3. BIT SIZE (BS) ──────────────────────────────────────────────────────
    elif c == "BS":
        return (
            "Nominal Bit Gauge Reference",
            f"Nominal drill bit size is {value:.2f} in. Used as geometric benchmark to quantify borehole enlargement (washout) and mudcake."
        )

    # ── 4. DEEP RESISTIVITY (RT) ──────────────────────────────────────────────
    elif c in ["RT", "ILD", "LLD", "RES", "RD"]:
        if value >= 50.0:
            return (
                "Optimal (Strong Hydrocarbon Indicator)",
                f"{value:.1f} Ohm.m is high. Non-conductive oil or gas fills the pore volume, preventing electrical current flow through saline brine. Prime pay zone candidate."
            )
        elif value >= 18.0:
            return (
                "Good (Commercial Resistivity Pay)",
                f"{value:.1f} Ohm.m indicates presence of hydrocarbons with minor capillary water cut. Economic pay interval."
            )
        elif value >= 6.0:
            return (
                "Moderate (Transition Zone / Low-Resistivity Pay)",
                f"{value:.1f} Ohm.m indicates oil-water transition zone, fresh formation water, or laminated shaly sand pay."
            )
        else:
            return (
                "Low Resistivity (Saline Aquifer / Conductive Shale)",
                f"{value:.1f} Ohm.m is low. Pores are dominated by highly conductive saline formation water or conductive clay minerals (water-bearing or shale)."
            )

    # ── 5. BULK DENSITY (RHOB) ────────────────────────────────────────────────
    elif c in ["RHOB", "RHO8", "DEN"]:
        if value < 2.05:
            return (
                "Abnormally Low Bulk Density",
                f"{value:.2f} g/cm3 is exceptionally low. Indicates high-porosity gas sand, coal bed, or severe borehole washout pad standoff."
            )
        elif value <= 2.28:
            return (
                "Optimal (High-Porosity Sandstone Matrix)",
                f"{value:.2f} g/cm3 corresponds to 22% - 35% porosity in quartz sandstone (rho_ma = 2.65 g/cm3). Excellent commercial storage capacity."
            )
        elif value <= 2.45:
            return (
                "Good (Moderate Commercial Porosity)",
                f"{value:.2f} g/cm3 corresponds to 12% - 22% porosity in sandstone. Economic formation storage."
            )
        else:
            return (
                "Dense / Tight Rock Matrix",
                f"{value:.2f} g/cm3 indicates low-porosity tight rock (< 10%), dense limestone, or heavy compacted shale."
            )

    # ── 6. DENSITY POROSITY (PHID / PHIE) ────────────────────────────────────
    elif c in ["PHID", "PHIE", "DPOR", "POR"]:
        pct = value * 100.0 if value <= 1.0 else value
        if pct >= 22.0:
            return (
                "Optimal (World-Class Reservoir Porosity)",
                f"{pct:.1f}% porosity represents world-class commercial pore volume (Brent/Wilcox benchmark: 20% - 28%). High storage capacity."
            )
        elif pct >= 14.0:
            return (
                "Good (Commercial Reservoir Porosity)",
                f"{pct:.1f}% porosity is good. Sufficient pore space to support commercial hydrocarbon reserves and drainage."
            )
        elif pct >= 8.0:
            return (
                "Moderate / Marginal Porosity",
                f"{pct:.1f}% porosity is fair to low. Marginal conventional reservoir, may require hydraulic fracture stimulation."
            )
        else:
            return (
                "Tight Matrix / Non-Reservoir",
                f"{pct:.1f}% porosity is tight. Sub-economic pore volume, non-reservoir rock, or tight seal."
            )

    # ── 7. NEUTRON POROSITY (NPHI) ───────────────────────────────────────────
    elif c in ["NPHI", "NPOR"]:
        phid = aux.get("PHID")
        pct = value * 100.0 if value <= 1.0 else value
        if phid is not None:
            phid_pct = phid * 100.0 if phid <= 1.0 else phid
            if pct < (phid_pct - 4.0):
                return (
                    "Gas Crossover Signature",
                    f"Neutron ({pct:.1f}%) drops below Density ({phid_pct:.1f}%). Classic gas effect: light gas hydrogen deficiency suppresses apparent neutron porosity."
                )
            elif pct > (phid_pct + 10.0):
                return (
                    "Shale Separation (Clay Bound Water)",
                    f"Neutron ({pct:.1f}%) significantly exceeds Density ({phid_pct:.1f}%). Clay minerals (illite/smectite) trap structural hydroxyl water that artificially inflates neutron readings."
                )
            else:
                return (
                    "Liquid Matrix Tracking",
                    f"Neutron ({pct:.1f}%) and Density ({phid_pct:.1f}%) track harmoniously, confirming liquid-filled (oil/water) clean formation."
                )
        else:
            if pct <= 15.0:
                return ("Low Apparent Neutron Porosity", f"Neutron porosity is {pct:.1f}%. Suggests low-porosity matrix or gas presence.")
            elif pct <= 30.0:
                return ("Optimal Liquid-Filled Sandstone", f"Neutron porosity ({pct:.1f}%) indicates clean, productive liquid-filled sandstone.")
            else:
                return ("High Neutron Porosity (Clay Effect)", f"Neutron porosity ({pct:.1f}%) is elevated due to clay-bound water.")

    # ── 8. WATER SATURATION (SW) ─────────────────────────────────────────────
    elif c in ["SW", "SWE"]:
        pct = value * 100.0 if value <= 1.0 else value
        hc_pct = max(0.0, 100.0 - pct)
        if pct <= 25.0:
            return (
                "Optimal (Pristine Hydrocarbon Pay)",
                f"Water saturation is only {pct:.1f}% (Hydrocarbon saturation: {hc_pct:.1f}%). Pristine commercial pay zone with minimal water production risk."
            )
        elif pct <= 45.0:
            return (
                "Good (Commercial Hydrocarbon Pay)",
                f"Water saturation is {pct:.1f}% (Hydrocarbon saturation: {hc_pct:.1f}%). Economic commercial pay with acceptable water cut."
            )
        elif pct <= 60.0:
            return (
                "Moderate (High Water Cut Risk)",
                f"Water saturation is {pct:.1f}% (Hydrocarbon saturation: {hc_pct:.1f}%). Transition zone; elevated risk of two-phase water/hydrocarbon flow."
            )
        else:
            return (
                "Water-Bearing (Sub-economic / Wet Zone)",
                f"Water saturation is {pct:.1f}% (Hydrocarbon cut: only {hc_pct:.1f}%). Water-dominated pore space; non-commercial zone."
            )

    # ── 9. HYDROCARBON SATURATION (SH) ───────────────────────────────────────
    elif c in ["SH", "SO", "HC"]:
        pct = value * 100.0 if value <= 1.0 else value
        if pct >= 70.0:
            return (
                "Optimal (High Hydrocarbon Saturation)",
                f"Hydrocarbon saturation is {pct:.1f}%. Substantial movable oil/gas column in pore space."
            )
        elif pct >= 50.0:
            return (
                "Good (Commercial Hydrocarbon Cut)",
                f"Hydrocarbon saturation is {pct:.1f}%. Commercial hydrocarbon pay."
            )
        else:
            return (
                "Sub-economic / Water Wet",
                f"Hydrocarbon saturation is only {pct:.1f}%. Water-dominated formation."
            )

    # ── 10. ACOUSTIC SONIC TRANSIT TIME (DT) ──────────────────────────────────
    elif c in ["DT", "DTP", "AC"]:
        if value <= 55.5:
            return (
                "Tight Rock Matrix (Low Acoustic Porosity)",
                f"{value:.1f} us/ft is fast transit time. Approaching pure sandstone quartz matrix (55.5 us/ft) or tight carbonate; minimal void space."
            )
        elif value <= 75.0:
            return (
                "Moderate Acoustic Porosity",
                f"{value:.1f} us/ft corresponds to 10% - 18% sonic porosity in sandstone."
            )
        elif value <= 95.0:
            return (
                "Optimal (High Acoustic Porosity Pay)",
                f"{value:.1f} us/ft corresponds to 18% - 30% sonic porosity. Well-sorted, permeable sandstone reservoir."
            )
        else:
            return (
                "Slow Transit Time (Soft Shale / Uncompacted)",
                f"{value:.1f} us/ft is very slow transit time. Typical of soft shales, shallow unconsolidated sands, or washouts."
            )

    # ── 11. SHALE VOLUME (VSH) ────────────────────────────────────────────────
    elif c in ["VSH", "VSH_LINEAR"]:
        pct = value * 100.0 if value <= 1.0 else value
        if pct <= 15.0:
            return (
                "Optimal (Clean Quartz Sand)",
                f"Shale volume is only {pct:.1f}%. Prime clean reservoir sand with high permeability and low clay content."
            )
        elif pct <= 35.0:
            return (
                "Good (Slightly Shaly Sand - Pay Conforming)",
                f"Shale volume is {pct:.1f}%. Conforms to commercial net pay criteria (Vsh <= 35%)."
            )
        else:
            return (
                "Non-Reservoir Shale Unit",
                f"Shale volume is {pct:.1f}% (exceeds 35% cutoff). Clay-clogged pores; excluded from commercial net pay."
            )

    # ── 12. FALLBACK DEFAULT ──────────────────────────────────────────────────
    return (
        "Logged Curve Measurement",
        f"Measured {c} = {value:.2f}. Within recorded well section."
    )


# ---------------------------------------------------------------------------
# Batch Customdata Generator for Plotly
# ---------------------------------------------------------------------------
def generate_hover_customdata(
    curve_name: str,
    values: pd.Series | np.ndarray,
    aux_dict: Optional[Dict[str, pd.Series | np.ndarray]] = None
) -> Tuple[np.ndarray, str]:
    """
    Generates a 2D numpy array of [status, interpretation] for every depth point,
    paired with an executive Plotly hovertemplate.
    
    Returns:
        (customdata_array, hovertemplate_string)
    """
    aux_dict = aux_dict or {}
    val_arr = np.asarray(values)
    n_points = len(val_arr)
    
    status_list = []
    interp_list = []
    
    # Pre-extract aux series arrays if available
    extracted_aux = {}
    for k, v in aux_dict.items():
        extracted_aux[k] = np.asarray(v) if v is not None else None
    
    for i in range(n_points):
        v = val_arr[i]
        pt_aux = {}
        for ak, av in extracted_aux.items():
            if av is not None and i < len(av):
                pt_aux[ak] = av[i]
        
        stat, interp = evaluate_point_status(curve_name, v, pt_aux)
        status_list.append(stat)
        interp_list.append(interp)
        
    customdata = np.stack((np.array(status_list, dtype=object), np.array(interp_list, dtype=object)), axis=-1)
    
    hovertemplate = (
        "<b>%{fullData.name}</b><br>"
        "Depth: <b>%{y:.1f} ft</b><br>"
        "Value: <b>%{x:.2f}</b><br>"
        "<b>Optimal Status:</b> %{customdata[0]}<br>"
        "<b>Meaning:</b> %{customdata[1]}"
        "<extra></extra>"
    )
    
    return customdata, hovertemplate


# ---------------------------------------------------------------------------
# Guide Metadata Retrieval Utilities
# ---------------------------------------------------------------------------
def get_curve_guide_for_well(available_curves: List[str]) -> List[Dict[str, Any]]:
    """
    Returns full reference guide metadata for all curves present in the active well.
    """
    guides = []
    for c in available_curves:
        c_upper = c.upper().strip()
        if c_upper in CURVE_KNOWLEDGE_BASE:
            meta = CURVE_KNOWLEDGE_BASE[c_upper].copy()
            meta["curve_symbol"] = c_upper
            guides.append(meta)
    return guides


def get_all_curve_guides() -> List[Dict[str, Any]]:
    """
    Returns full reference guide metadata for all canonical wireline curves.
    """
    guides = []
    for c, meta in CURVE_KNOWLEDGE_BASE.items():
        m = meta.copy()
        m["curve_symbol"] = c
        guides.append(m)
    return guides


def get_guide_comparison_dataframe(available_curves: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Formats a clean reference table comparing all curves, their physical min/max bounds,
    ideal values, shale baselines, operational cutoffs, and literature citations.
    """
    if available_curves:
        # Match available curves first, then append derived petrophysical curves
        target_keys = [c.upper().strip() for c in available_curves if c.upper().strip() in CURVE_KNOWLEDGE_BASE]
        for derived in ["VSH", "PHID", "SW"]:
            if derived not in target_keys and derived in CURVE_KNOWLEDGE_BASE:
                target_keys.append(derived)
    else:
        target_keys = list(CURVE_KNOWLEDGE_BASE.keys())

    rows = []
    for k in target_keys:
        meta = CURVE_KNOWLEDGE_BASE[k]
        min_v = meta.get("min_value")
        max_v = meta.get("max_value")
        unit = meta.get("unit", "")
        rows.append({
            "Curve": k,
            "Measurement Name": meta["name"],
            "Unit": unit,
            "Physical Min": f"{min_v} {unit}" if min_v is not None else "N/A",
            "Physical Max": f"{max_v} {unit}" if max_v is not None else "N/A",
            "Ideal Commercial Reservoir Range": meta["ideal_value"],
            "Shale / Non-Pay Baseline": meta["shale_baseline"],
            "Pay Cutoff Rule": meta["cutoff_value"],
            "Authoritative Literature Source": meta["reference"],
        })
    return pd.DataFrame(rows)


def show_guidelines_dialog(available_curves: Optional[List[str]] = None):
    """
    Renders the full formation evaluation guidelines inside a Streamlit dialog modal.
    Lists all physical min and max values, ideal ranges, cutoffs, and engineering principles.
    """
    import streamlit as st
    from welllogqc.app.ui_theme import clean_html

    @st.dialog("Formation Evaluation Guidelines & Log Standards", width="large")
    def _guidelines_modal():
        st.markdown(clean_html("""
        <div style="background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 0 6px 6px 0; padding: 12px 18px; margin-bottom: 14px;">
            <div style="color: #38bdf8; font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">
                Petrophysical Engineering Manual & Sensor Physical Limits
            </div>
            <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.5;">
                Every downhole wireline sensor operates on fundamental physics (nuclear scintillation, electron scattering, hydrogen index moderation, induction resistivity, and acoustic wave transit). 
                Below are the physical minimum and maximum limits, ideal commercial baselines, and pay cutoff rules compiled from Schlumberger, AAPG, and SPWLA standards.
            </div>
        </div>
        """), unsafe_allow_html=True)

        all_keys = list(CURVE_KNOWLEDGE_BASE.keys())
        if available_curves:
            avail_upper = [c.upper().strip() for c in available_curves if c.upper().strip() in CURVE_KNOWLEDGE_BASE]
            other_keys = [k for k in all_keys if k not in avail_upper]
            options_keys = avail_upper + other_keys
        else:
            options_keys = all_keys

        curve_labels = [f"{k} - {CURVE_KNOWLEDGE_BASE[k]['name']}" for k in options_keys]
        selected_idx = st.selectbox(
            "Select Wireline Log to Inspect",
            range(len(options_keys)),
            format_func=lambda i: curve_labels[i],
            key="modal_guidelines_log_select"
        )
        selected_key = options_keys[selected_idx]
        meta = CURVE_KNOWLEDGE_BASE[selected_key]
        min_v = meta.get("min_value", "N/A")
        max_v = meta.get("max_value", "N/A")
        unit = meta.get("unit", "")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(clean_html(f"""
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 14px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <strong style="color: #38bdf8; font-size: 1.05rem;">{selected_key} - {meta['name']}</strong>
                    <span style="background: #1e293b; color: #cbd5e1; padding: 2px 8px; border-radius: 4px; border: 1px solid #334155; font-size: 0.8rem;">{unit}</span>
                </div>
                <div style="background: #1e293b; border-radius: 4px; padding: 8px; margin-bottom: 10px; border: 1px solid #334155;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px;">
                        <span style="color: #94a3b8;">Physical Minimum:</span>
                        <span style="color: #10b981; font-weight: 700;">{min_v} {unit}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.82rem;">
                        <span style="color: #94a3b8;">Physical Maximum:</span>
                        <span style="color: #ef4444; font-weight: 700;">{max_v} {unit}</span>
                    </div>
                </div>
                <strong style="color: #38bdf8; font-size: 0.86rem;">Ideal Commercial Reservoir Value:</strong><br>
                <span style="color: #f8fafc; font-size: 0.90rem; font-weight: 600;">{meta['ideal_value']}</span><br><br>
                <strong style="color: #10b981; font-size: 0.86rem;">Pay Cutoff Rule:</strong><br>
                <span style="color: #cbd5e1; font-size: 0.86rem;">{meta['cutoff_value']}</span><br><br>
                <strong style="color: #f59e0b; font-size: 0.86rem;">Shale / Non-Pay Baseline:</strong><br>
                <span style="color: #cbd5e1; font-size: 0.86rem;">{meta['shale_baseline']}</span>
            </div>
            """), unsafe_allow_html=True)
        with col2:
            st.markdown(clean_html(f"""
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 14px; margin-bottom: 10px;">
                <strong style="color: #38bdf8; font-size: 0.86rem;">Physical Measurement Principle:</strong><br>
                <span style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.45;">{meta['physics']}</span><br><br>
                <strong style="color: #38bdf8; font-size: 0.86rem;">Petrophysical Significance:</strong><br>
                <span style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.45;">{meta['significance']}</span>
            </div>
            """), unsafe_allow_html=True)

        st.markdown(clean_html(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 6px; margin-bottom: 8px;">
            <div style="background: #0f172a; border-left: 4px solid #ef4444; padding: 10px 14px; border-radius: 0 6px 6px 0;">
                <strong style="color: #ef4444; font-size: 0.84rem;">What High Readings Mean:</strong><br>
                <span style="color: #cbd5e1; font-size: 0.82rem; line-height: 1.4;">{meta['high_meaning']}</span>
            </div>
            <div style="background: #0f172a; border-left: 4px solid #10b981; padding: 10px 14px; border-radius: 0 6px 6px 0;">
                <strong style="color: #10b981; font-size: 0.84rem;">What Low Readings Mean:</strong><br>
                <span style="color: #cbd5e1; font-size: 0.82rem; line-height: 1.4;">{meta['low_meaning']}</span>
            </div>
        </div>
        <div style="background: #0f172a; border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 0 6px 6px 0; margin-top: 6px;">
            <strong style="color: #f59e0b; font-size: 0.84rem;">Operational Traps & Failure Modes (QC Alert):</strong><br>
            <span style="color: #cbd5e1; font-size: 0.82rem; line-height: 1.4;">{meta['common_pitfalls']}</span>
        </div>
        <div style="color: #64748b; font-size: 0.78rem; margin-top: 8px;">
            <strong>Literature Citation:</strong> {meta['reference']}
        </div>
        """), unsafe_allow_html=True)

        with st.expander("Master Physical Bounds Table (All 24 Standard Logs)", expanded=False):
            min_max_rows = []
            for k in all_keys:
                m = CURVE_KNOWLEDGE_BASE[k]
                min_max_rows.append({
                    "Log": k,
                    "Name": m["name"],
                    "Physical Min": f"{m.get('min_value')} {m.get('unit')}",
                    "Physical Max": f"{m.get('max_value')} {m.get('unit')}",
                    "Ideal Range": m.get("ideal_value"),
                    "Cutoff": m.get("cutoff_value")
                })
            st.dataframe(pd.DataFrame(min_max_rows), use_container_width=True, hide_index=True)

    _guidelines_modal()


def render_sidebar_guidance(available_curves: Optional[List[str]] = None):
    """
    Renders an authoritative Petrophysical Reference Guidance panel in the Streamlit left sidebar.
    Lists all formation evaluation logs documented in literature with their:
    - Physical Min and Max values
    - Unit of measurement
    - Ideal commercial range
    - Standard cutoffs
    - Physical principle and engineering interpretation
    """
    import streamlit as st
    from welllogqc.app.ui_theme import ICONS, clean_html, render_sidebar_header
    
    render_sidebar_header("Guidance: Formation Logs", "compass")
    
    st.sidebar.markdown(
        "<div style='color: #94a3b8; font-size: 0.82rem; line-height: 1.4; margin-bottom: 10px;'>"
        "Authoritative reference guide with physical minimum & maximum bounds, ideal commercial ranges, "
        "and cutoffs for all formation evaluation logs."
        "</div>",
        unsafe_allow_html=True
    )
    
    all_keys = list(CURVE_KNOWLEDGE_BASE.keys())
    
    # Priority sorting: if available_curves given, place active well curves on top
    if available_curves:
        avail_upper = [c.upper().strip() for c in available_curves if c.upper().strip() in CURVE_KNOWLEDGE_BASE]
        other_keys = [k for k in all_keys if k not in avail_upper]
        options_keys = avail_upper + other_keys
    else:
        options_keys = all_keys
        
    curve_labels = [f"{k} - {CURVE_KNOWLEDGE_BASE[k]['name']}" for k in options_keys]
    selected_idx = st.sidebar.selectbox(
        "Select Log for Guidance",
        range(len(options_keys)),
        format_func=lambda i: curve_labels[i],
        key="sidebar_guidance_log_select"
    )
    
    selected_key = options_keys[selected_idx]
    meta = CURVE_KNOWLEDGE_BASE[selected_key]
    
    min_v = meta.get("min_value", "N/A")
    max_v = meta.get("max_value", "N/A")
    unit = meta.get("unit", "")
    
    st.sidebar.markdown(clean_html(f"""
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 12px; margin-top: 6px; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <strong style="color: #38bdf8; font-size: 0.95rem;">{selected_key}</strong>
            <span style="background: #0f172a; color: #cbd5e1; font-size: 0.76rem; padding: 2px 8px; border-radius: 4px; border: 1px solid #334155;">{unit}</span>
        </div>
        <div style="color: #f8fafc; font-size: 0.84rem; font-weight: 600; margin-bottom: 6px;">{meta['name']}</div>
        <div style="background: #0f172a; border-radius: 4px; padding: 8px; margin-bottom: 8px; border: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 3px;">
                <span style="color: #94a3b8;">Physical Min:</span>
                <span style="color: #10b981; font-weight: 700;">{min_v} {unit}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem;">
                <span style="color: #94a3b8;">Physical Max:</span>
                <span style="color: #ef4444; font-weight: 700;">{max_v} {unit}</span>
            </div>
        </div>
        <div style="font-size: 0.80rem; margin-bottom: 6px;">
            <span style="color: #38bdf8; font-weight: 600;">Category:</span>
            <span style="color: #cbd5e1;"> {meta['category']}</span>
        </div>
        <div style="font-size: 0.80rem; margin-bottom: 6px;">
            <span style="color: #10b981; font-weight: 600;">Ideal Reservoir:</span>
            <span style="color: #cbd5e1;"> {meta['ideal_value']}</span>
        </div>
        <div style="font-size: 0.80rem; margin-bottom: 6px;">
            <span style="color: #f59e0b; font-weight: 600;">Pay Cutoff:</span>
            <span style="color: #cbd5e1;"> {meta['cutoff_value']}</span>
        </div>
        <div style="font-size: 0.80rem; margin-bottom: 6px;">
            <span style="color: #94a3b8; font-weight: 600;">Non-Pay / Shale:</span>
            <span style="color: #cbd5e1;"> {meta['shale_baseline']}</span>
        </div>
        <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.4; border-top: 1px solid #334155; padding-top: 6px; margin-top: 6px;">
            <strong>Physics:</strong> {meta['physics']}
        </div>
    </div>
    """), unsafe_allow_html=True)
    
    with st.sidebar.expander("Physical Min/Max Quick Table", expanded=False):
        min_max_rows = []
        for k in all_keys:
            m = CURVE_KNOWLEDGE_BASE[k]
            min_max_rows.append({
                "Curve": k,
                "Min": f"{m.get('min_value')} {m.get('unit')}",
                "Max": f"{m.get('max_value')} {m.get('unit')}",
            })
        st.dataframe(pd.DataFrame(min_max_rows), use_container_width=True, hide_index=True)

