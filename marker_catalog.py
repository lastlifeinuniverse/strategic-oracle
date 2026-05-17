"""
Marker Catalog  canonical lab marker codes, units, and reference ranges.

Two range tiers per marker:
  normal     standard clinical reference range (lab-reported)
  optimal    tighter longevity/preventive ranges from EAS 2022, ACC/AHA, SG MOH CPGs

Flag rules (applied in order):
  'abnormal'    outside normal reference range
  'suboptimal'  inside normal but outside optimal longevity range
  'optimal'     within longevity-optimal range
"""

from __future__ import annotations
from typing import Optional

# ---------------------------------------------------------------------------
# Canonical marker definitions
# ---------------------------------------------------------------------------
# Each entry: (name, unit, normal_low, normal_high, optimal_low, optimal_high)
# None means "no bound on that side"
#
# Sources:
#   EAS 2022 Lp(a) consensus, ACC/AHA 2019 lipids, SG MOH 2023 diabetes CPG,
#   Endocrine Society VitD guidelines, KDIGO eGFR staging

MARKER_CATALOG: dict[str, dict] = {
    "LDL_C": {
        "name": "LDL Cholesterol",
        "unit": "mmol/L",
        "normal":  (None, 3.40),
        "optimal": (None, 1.80),   # EAS 2022 high-risk target
    },
    "HDL_C": {
        "name": "HDL Cholesterol",
        "unit": "mmol/L",
        "normal":  (1.00, None),
        "optimal": (1.50, None),
    },
    "TC": {
        "name": "Total Cholesterol",
        "unit": "mmol/L",
        "normal":  (None, 5.20),
        "optimal": (None, 4.50),
    },
    "TG": {
        "name": "Triglycerides",
        "unit": "mmol/L",
        "normal":  (None, 1.70),
        "optimal": (None, 1.10),
    },
    "LP_A": {
        "name": "Lipoprotein(a)",
        "unit": "nmol/L",
        "normal":  (None, 125.0),  # most lab cut-offs
        "optimal": (None, 75.0),   # EAS 2022 high-risk threshold
    },
    "APO_B": {
        "name": "Apolipoprotein B",
        "unit": "g/L",
        "normal":  (None, 1.00),
        "optimal": (None, 0.80),
    },
    "APO_A1": {
        "name": "Apolipoprotein A1",
        "unit": "g/L",
        "normal":  (1.00, None),
        "optimal": (1.40, None),
    },
    "HOMOCYSTEINE": {
        "name": "Homocysteine",
        "unit": "mol/L",
        "normal":  (None, 15.0),
        "optimal": (None, 10.0),
    },
    "HBA1C": {
        "name": "HbA1c",
        "unit": "%",
        "normal":  (None, 5.70),
        "optimal": (None, 5.30),
    },
    "GLUCOSE_FASTING": {
        "name": "Fasting Glucose",
        "unit": "mmol/L",
        "normal":  (None, 6.10),
        "optimal": (None, 5.00),
    },
    "HS_CRP": {
        "name": "hs-CRP",
        "unit": "mg/L",
        "normal":  (None, 5.00),
        "optimal": (None, 1.00),   # cardiovascular low-risk threshold
    },
    "TSH": {
        "name": "TSH",
        "unit": "mIU/L",
        "normal":  (0.40, 4.00),
        "optimal": (0.50, 2.50),
    },
    "FT4": {
        "name": "Free T4",
        "unit": "pmol/L",
        "normal":  (9.00, 25.00),
        "optimal": (12.0, 22.0),
    },
    "VIT_D": {
        "name": "25-OH Vitamin D",
        "unit": "nmol/L",
        "normal":  (50.0, None),
        "optimal": (100.0, 150.0),
    },
    "B12": {
        "name": "Vitamin B12",
        "unit": "pmol/L",
        "normal":  (145.0, None),
        "optimal": (300.0, None),
    },
    "FOLATE": {
        "name": "Folate",
        "unit": "nmol/L",
        "normal":  (7.00, None),
        "optimal": (20.0, None),
    },
    "FERRITIN": {
        "name": "Ferritin",
        "unit": "g/L",
        "normal":  (13.0, 150.0),  # female; male upper ~300
        "optimal": (30.0, 100.0),
    },
    "EGFR": {
        "name": "eGFR",
        "unit": "mL/min/1.73m",
        "normal":  (60.0, None),
        "optimal": (90.0, None),
    },
    "VO2_MAX": {
        "name": "VO2max",
        "unit": "mL/kg/min",
        # ACSM norms for 40-49yo male: Very Poor <31.5, Poor 31.5-35.3, Fair 35.4-40.0, Good 40.1-44.9, Excellent 45.0-49.4, Superior ≥49.5
        "normal":  (31.5, None),   # below "Poor" threshold for 40-49yo male
        "optimal": (45.0, None),   # Excellent range for 40-49yo male
    },
    "VO2_MAX_PERCENTILE": {
        "name": "VO2max Percentile",
        "unit": "%ile",
        "normal":  (20.0, None),
        "optimal": (60.0, None),
    },
    "VT1_HR": {
        "name": "Ventilatory Threshold 1 HR",
        "unit": "bpm",
        "normal":  (None, None),
        "optimal": (None, None),
    },
    "VT2_HR": {
        "name": "Ventilatory Threshold 2 HR",
        "unit": "bpm",
        "normal":  (None, None),
        "optimal": (None, None),
    },
    "VT1_POWER": {
        "name": "VT1 Power",
        "unit": "W",
        "normal":  (None, None),
        "optimal": (None, None),
    },
    "VT2_POWER": {
        "name": "VT2 Power",
        "unit": "W",
        "normal":  (None, None),
        "optimal": (None, None),
    },
    "MAX_HR_MEASURED": {
        "name": "Max Heart Rate (Measured)",
        "unit": "bpm",
        "normal":  (None, None),
        "optimal": (None, None),
    },
    "MAX_POWER": {
        "name": "Max Power (VO2max Test)",
        "unit": "W",
        "normal":  (None, None),
        "optimal": (None, None),
    },
    "GXT_ZONE1_HR_MAX": {
        "name": "Training Zone 1 Max HR",
        "unit": "bpm",
        "normal":  (None, None),
        "optimal": (None, None),
    },
    "GXT_ZONE2_HR_MAX": {
        "name": "Training Zone 2 Max HR",
        "unit": "bpm",
        "normal":  (None, None),
        "optimal": (None, None),
    },
    "GXT_ZONE3_HR_MAX": {
        "name": "Training Zone 3 Max HR",
        "unit": "bpm",
        "normal":  (None, None),
        "optimal": (None, None),
    },
    "GXT_ZONE4_HR_MAX": {
        "name": "Training Zone 4 Max HR",
        "unit": "bpm",
        "normal":  (None, None),
        "optimal": (None, None),
    },
    "FIB4": {
        "name": "FIB-4 Liver Score",
        "unit": "",
        "normal":  (None, 1.30),
        "optimal": (None, 1.00),
    },
    "CTDNA_MCED": {
        "name": "SPOT-MAS Multi-Cancer Score",
        "unit": "z-score",
        "normal":  (None, 2.00),
        "optimal": (None, 1.00),
    },
}

# ---------------------------------------------------------------------------
# Name  code mapping (lower-cased, stripped)
# Covers common variations from Innoquest, SPOT-MAS, VO2 Master reports
# ---------------------------------------------------------------------------
NAME_TO_CODE: dict[str, str] = {
    # Lipids
    "ldl cholesterol":                      "LDL_C",
    "ldl-cholesterol":                      "LDL_C",
    "low density lipoprotein":              "LDL_C",
    "hdl cholesterol":                      "HDL_C",
    "hdl-cholesterol":                      "HDL_C",
    "high density lipoprotein":             "HDL_C",
    "total cholesterol":                    "TC",
    "cholesterol":                          "TC",
    "triglycerides":                        "TG",
    "triglyceride":                         "TG",
    # Special lipids
    "lipoprotein (a)":                      "LP_A",
    "lipoprotein(a)":                       "LP_A",
    "lp(a)":                                "LP_A",
    "lp (a)":                               "LP_A",
    "apolipoprotein b":                     "APO_B",
    "apo b":                                "APO_B",
    "apolipoprotein a1":                    "APO_A1",
    "apolipoprotein a-1":                   "APO_A1",
    "apo a1":                               "APO_A1",
    # Cardiac risk
    "homocysteine":                         "HOMOCYSTEINE",
    "high sensitivity c-reactive protein":  "HS_CRP",
    "hs-crp":                               "HS_CRP",
    "hs crp":                               "HS_CRP",
    "hsCRP":                                "HS_CRP",
    # Diabetes
    "hba1c":                                "HBA1C",
    "hb a1c":                               "HBA1C",
    "glycated haemoglobin":                 "HBA1C",
    "glycated hemoglobin":                  "HBA1C",
    "fasting glucose":                      "GLUCOSE_FASTING",
    "glucose, fasting":                     "GLUCOSE_FASTING",
    "blood glucose":                        "GLUCOSE_FASTING",
    # Thyroid
    "tsh":                                  "TSH",
    "thyroid stimulating hormone":          "TSH",
    "free t4":                              "FT4",
    "ft4":                                  "FT4",
    "free thyroxine":                       "FT4",
    # Vitamins / minerals
    "25-oh vitamin d":                      "VIT_D",
    "25-hydroxyvitamin d":                  "VIT_D",
    "vitamin d":                            "VIT_D",
    "vitamin d3":                           "VIT_D",
    "vitamin b12":                          "B12",
    "b12":                                  "B12",
    "cobalamin":                            "B12",
    "folate":                               "FOLATE",
    "folic acid":                           "FOLATE",
    "ferritin":                             "FERRITIN",
    # Kidney
    "egfr":                                 "EGFR",
    "estimated gfr":                        "EGFR",
    "estimated glomerular filtration rate": "EGFR",
    # VO2 / fitness
    "vo2max":                               "VO2_MAX",
    "vo2 max":                              "VO2_MAX",
    "vomax":                               "VO2_MAX",
    "vo₂max":                               "VO2_MAX",
    "maximal oxygen uptake":                "VO2_MAX",
    "vo2max percentile":                    "VO2_MAX_PERCENTILE",
    "ventilatory threshold 1 hr":           "VT1_HR",
    "ventilatory threshold 1 heart rate":   "VT1_HR",
    "ventilatory threshold 1":              "VT1_HR",
    "vt1 heart rate":                       "VT1_HR",
    "vt1 hr":                               "VT1_HR",
    "vt1":                                  "VT1_HR",
    "ventilatory threshold 2 hr":           "VT2_HR",
    "ventilatory threshold 2 heart rate":   "VT2_HR",
    "ventilatory threshold 2":              "VT2_HR",
    "vt2 heart rate":                       "VT2_HR",
    "vt2 hr":                               "VT2_HR",
    "vt2":                                  "VT2_HR",
    "vt1 power":                            "VT1_POWER",
    "vt2 power":                            "VT2_POWER",
    "max power":                            "MAX_POWER",
    "maximum power":                        "MAX_POWER",
    "max heart rate measured":              "MAX_HR_MEASURED",
    "measured max hr":                      "MAX_HR_MEASURED",
    "training zone 1 max hr":              "GXT_ZONE1_HR_MAX",
    "training zone 2 max hr":              "GXT_ZONE2_HR_MAX",
    "training zone 3 max hr":              "GXT_ZONE3_HR_MAX",
    "training zone 4 max hr":              "GXT_ZONE4_HR_MAX",
    # Liver / cancer
    "fib-4":                                "FIB4",
    "fib4":                                 "FIB4",
    "spot-mas":                             "CTDNA_MCED",
    "mced score":                           "CTDNA_MCED",
    "multi-cancer early detection":         "CTDNA_MCED",
}


def resolve_marker_code(raw_name: str) -> Optional[str]:
    """Return canonical marker code for a raw test name, or None if unknown."""
    key = raw_name.lower().strip()
    # Direct lookup
    if key in NAME_TO_CODE:
        return NAME_TO_CODE[key]
    # Partial match  find the first catalog entry whose key is a substring
    for alias, code in NAME_TO_CODE.items():
        if alias in key or key in alias:
            return code
    return None


def compute_flag(code: str, value: float) -> str:
    """Return 'optimal' | 'suboptimal' | 'normal' | 'abnormal' for a marker value."""
    entry = MARKER_CATALOG.get(code)
    if not entry:
        return "normal"

    def _in(v: float, lo: Optional[float], hi: Optional[float]) -> bool:
        if lo is not None and v < lo:
            return False
        if hi is not None and v > hi:
            return False
        return True

    norm_lo, norm_hi   = entry["normal"]
    opt_lo,  opt_hi    = entry["optimal"]

    if not _in(value, norm_lo, norm_hi):
        return "abnormal"
    if opt_lo is None and opt_hi is None:
        return "normal"
    if _in(value, opt_lo, opt_hi):
        return "optimal"
    return "suboptimal"
