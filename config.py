DEFAULT_PARAMS = {
    "session_max_s": 600.0,
    "min_ili_ms": 60.0,
    "burst_threshold_ms": 330.0,
    "min_licks_per_burst": 3,
    "hist_bin_ms": 5.0,
    "hist_range_min_ms": 60.0,
    "hist_range_max_ms": 500.0,
    "low_range_min_ms": 60.0,
    "low_range_max_ms": 220.0,
    "high_range_min_ms": 220.0,
    "high_range_max_ms": 330.0,
}

GROUP_MAPS = {
    "BSC": {"VEH": {"group": "Veh", "order": 1}, "SCOP": {"group": "SCOP intra-BLA", "order": 2}},
    "OXO": {
        "VEH": {"group": "Veh", "dose": 0.0, "order": 1},
        "O812": {"group": "OXO 8.12", "dose": 8.12, "order": 2},
        "O227": {"group": "OXO 22.7", "dose": 22.7, "order": 3},
        "O60": {"group": "OXO 60", "dose": 60.0, "order": 4},
    },
    "OSIP": {
        "VV": {"group": "Veh BLA + Veh IP", "drug_bla": "Veh", "drug_ip": "Veh", "order": 1},
        "OV": {"group": "OXO BLA + Veh IP", "drug_bla": "OXO", "drug_ip": "Veh", "order": 2},
        "VS": {"group": "Veh BLA + SCOP HB IP", "drug_bla": "Veh", "drug_ip": "SCOP HB", "order": 3},
        "OS": {"group": "OXO BLA + SCOP HB IP", "drug_bla": "OXO", "drug_ip": "SCOP HB", "order": 4},
    },
    "HB": {
        "VEH": {"group": "Veh", "order": 1},
        ".3": {"group": "SCOP HB 0.3", "dose_mgkg": 0.3, "order": 2},
        "1": {"group": "SCOP HB 1", "dose_mgkg": 1.0, "order": 3},
        "3": {"group": "SCOP HB 3", "dose_mgkg": 3.0, "order": 4},
    },
    "NB": {
        "VEH": {"group": "Veh", "order": 1},
        ".3": {"group": "SCOP NB 0.3", "dose_mgkg": 0.3, "order": 2},
        "1": {"group": "SCOP NB 1", "dose_mgkg": 1.0, "order": 3},
        "3": {"group": "SCOP NB 3", "dose_mgkg": 3.0, "order": 4},
    },
    "SCL": {
        "VEH": {"group": "Veh", "order": 1},
        "NB": {"group": "SCOP NB", "order": 2},
        "HB": {"group": "SCOP HB", "order": 3},
        "LICL": {"group": "LiCl", "order": 4},
    },
}
SKIP_SHEETS = {"metadata", "notas", "resumen", "readme", "graphpad", "figuras"}
