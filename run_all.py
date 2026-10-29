"""Run the whole pipeline end to end, from the extracted REFIT files to every
table in results/ and figure in figures/.

    python run_all.py

Steps: prepare CLEAN houses -> Section 1 (raw House 1) -> Section 2 (EDA) ->
Section 3 (model). About 5-8 minutes on a 4-core laptop.
"""
import time

from src import config, prepare_clean, section1_raw, section2_eda, section3_model
from src.refit_io import clean_house_file, raw_house_files

STEPS = [
    ("Prepare CLEAN washing-machine houses (1-minute parquet + screening)", prepare_clean.main),
    ("Section 1: inspect and clean RAW House 1", section1_raw.main),
    ("Section 2: washing-machine EDA", section2_eda.main),
    ("Section 3: disaggregation model and evaluation", section3_model.main),
]


def check_data():
    missing = []
    for find, house in [(raw_house_files, 1)] + [(clean_house_file, h) for h in config.WASHING_MACHINE_CHANNEL]:
        try:
            find(house)
        except FileNotFoundError as e:
            missing.append(e)
    if missing:
        raise SystemExit(
            "REFIT data not found. Download and extract both archives into data/archives/ "
            "as described in README.md (Data). Missing, e.g.: " + str(missing[0]))


def main():
    check_data()
    start = time.time()
    for name, step in STEPS:
        t = time.time()
        print(f"\n=== {name} ===")
        step()
        print(f"--- {name}: {time.time() - t:.0f} s")
    print(f"\nAll done in {(time.time() - start) / 60:.1f} min. "
          f"Tables in {config.RESULTS}, figures in {config.FIGURES}.")


if __name__ == "__main__":
    main()
