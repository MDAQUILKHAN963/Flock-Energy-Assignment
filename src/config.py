"""Project-wide paths and dataset metadata.

Analysis thresholds live next to the code that justifies them (see each
section's module); only facts taken from the dataset documentation are here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
ARCHIVES = DATA / "archives"                    # the two .7z files, extracted in place
RAW_DIR = ARCHIVES / "REFIT_RAW_081116"         # RAW_House1_Part1.csv, ...
CLEAN_DIR = ARCHIVES / "CLEAN_REFIT_081116"     # CLEAN_House1.csv, ...
PROCESSED = DATA / "processed"                  # 1-minute parquet files produced by the pipeline

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

# House 1 channel labels, per docs/CLEAN_READ_ME_081116.txt.
HOUSE1_CHANNELS = {
    "Aggregate": "Aggregate",
    "Appliance1": "Fridge",
    "Appliance2": "Freezer (1)",
    "Appliance3": "Freezer (2)",
    "Appliance4": "Washer dryer",
    "Appliance5": "Washing machine",
    "Appliance6": "Dishwasher",
    "Appliance7": "Computer",
    "Appliance8": "Television site",
    "Appliance9": "Electric heater",
}

# Houses used in Sections 2-3 (DATA.md section 6). Excluded: 3, 11, 21 (solar
# PV in the aggregate), 4 (two washing machines), 13 (machine replaced mid-study).
STUDY_HOUSES = [1, 2, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20]

# Washing-machine channel per house, per docs/CLEAN_READ_ME_081116.txt.
# House 4 is omitted (two washing machines on separate channels) and
# Houses 12 and 14 have no washing-machine channel.
WASHING_MACHINE_CHANNEL = {
    1: "Appliance5",
    2: "Appliance2",
    3: "Appliance6",
    5: "Appliance3",
    6: "Appliance2",
    7: "Appliance5",
    8: "Appliance4",
    9: "Appliance3",
    10: "Appliance5",
    11: "Appliance3",
    13: "Appliance3",
    15: "Appliance3",
    16: "Appliance5",
    17: "Appliance4",
    18: "Appliance5",
    19: "Appliance2",
    20: "Appliance4",
    21: "Appliance3",
}
