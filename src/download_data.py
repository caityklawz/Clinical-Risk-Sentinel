"""
Downloads and prepares both datasets for the Clinical Risk Sentinel project.

Datasets (both public / de-identified):
1. Diabetes 130-US Hospitals (1999-2008) — UCI ML Repository, via GitHub mirror
   Source: Clore, Cios, DeShazo, Strack (2014), UCI ML Repository
   https://doi.org/10.24432/C5230J
2. PhysioNet/CinC Challenge 2019 — Early Prediction of Sepsis from Clinical Data
   Source: Reyna et al. (2019), PhysioNet
   https://physionet.org/content/challenge-2019/1.0.0/

Run this once before anything else:
    python src/download_data.py
"""

import os
import glob
import zipfile
import shutil
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TMP_DIR = os.path.join(DATA_DIR, "_tmp_sepsis_extract")


def download_file(url: str, dest_path: str) -> None:
    print(f"  Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Saved to {dest_path} ({os.path.getsize(dest_path) / 1e6:.1f} MB)")


def get_diabetes_data() -> None:
    print("\n[1/2] Diabetes 130-US Hospitals dataset")
    os.makedirs(DATA_DIR, exist_ok=True)

    diabetic_data_url = (
        "https://raw.githubusercontent.com/andrewwlong/diabetes_readmission/"
        "master/diabetic_data.csv"
    )
    ids_mapping_url = (
        "https://raw.githubusercontent.com/andrewwlong/diabetes_readmission/"
        "master/IDs_mapping.csv"
    )

    download_file(diabetic_data_url, os.path.join(DATA_DIR, "diabetic_data.csv"))
    download_file(ids_mapping_url, os.path.join(DATA_DIR, "IDs_mapping.csv"))


def get_sepsis_data() -> None:
    print("\n[2/2] PhysioNet Sepsis Challenge dataset")
    combined_path = os.path.join(DATA_DIR, "sepsis_combined.csv")

    if os.path.exists(combined_path):
        print(f"  Already exists at {combined_path}, skipping.")
        return

    os.makedirs(TMP_DIR, exist_ok=True)
    zip_path = os.path.join(TMP_DIR, "sepsis_repo.zip")

    repo_zip_url = (
        "https://codeload.github.com/MartinOravecSvK/"
        "Early-Prediction-of-Sepsis/zip/refs/heads/main"
    )
    print("  This is ~900MB and may take a few minutes...")
    download_file(repo_zip_url, zip_path)

    print("  Extracting .psv files...")
    with zipfile.ZipFile(zip_path, "r") as z:
        members = [
            m for m in z.namelist()
            if "/Dataset/training_set" in m and m.endswith(".psv")
        ]
        z.extractall(TMP_DIR, members=members)

    print("  Combining patient files into a single CSV (this takes a minute)...")
    base = os.path.join(TMP_DIR, "Early-Prediction-of-Sepsis-main", "Dataset")
    dfs = []
    for set_name, hospital_label in [("training_setA", "A"), ("training_setB", "B")]:
        files = sorted(glob.glob(os.path.join(base, set_name, "*.psv")))
        print(f"    {set_name}: {len(files)} patient files")
        for f in files:
            patient_id = os.path.basename(f).replace(".psv", "")
            df = pd.read_csv(f, sep="|")
            df.insert(0, "PatientID", patient_id)
            df.insert(1, "HospitalSystem", hospital_label)
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(combined_path, index=False)
    print(f"  Saved combined dataset: {combined.shape} -> {combined_path}")
    print(f"  Unique patients: {combined['PatientID'].nunique()}")
    print(f"  Sepsis-positive rows: {combined['SepsisLabel'].sum()} "
          f"({100 * combined['SepsisLabel'].mean():.2f}%)")

    print("  Cleaning up temporary extracted files...")
    shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    get_diabetes_data()
    get_sepsis_data()
    print("\nAll datasets ready in ./data/")
