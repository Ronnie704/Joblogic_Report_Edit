import io
import os
from ftplib import FTP_TLS, error_perm
import pandas as pd

FTP_HOST = "ftp.drivehq.com"
FTP_USER = os.environ.get("FTP_USER")
FTP_PASS = os.environ.get("FTP_PASS")

INPUT_DIR = "/JoblogicReports"
OUTPUT_DIR = "/JoblogicReports/processed"


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up the Joblogic export:
    - drop unwanted columns
    - PPM: if Total Sell == 0, use Job Ref 1 as Total Sell
    - Quoted: Total Sell = Total Sell - Material Sell
    - Labour = Total Sell - Material Sell
    - sort by Engineer
    """

    # 1. Drop columns
    columns_to_drop = [
        "Job Ref 2",
        "Expense Cost",
        "Expense Sell",
        "Reference Number",
        "Quoted Number",   # NOTE: if your column is actually "Quote Number", change this
        "Completed Date",
    ]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

    # 2. Convert numeric columns
    for col in ["Total Sell", "Material Sell", "Job Ref 1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True),
                errors="coerce",
            )

    # 2b convert datetime columns
    for col in ["Job Travel", "Time on Site", "Time off Site", "Home Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col].astype(str),
                dayfirst=True,
                errors="coerce",
            )

    # 3. If Job Type == 'PPM' and Total Sell == 0, use Job Ref 1 as new Total Sell
    if {"Job Type", "Total Sell", "Job Ref 1"}.issubset(df.columns):
        condition_ppm = (
            df["Job Type"].str.strip().str.upper() == "PPM"
        ) & (df["Total Sell"].fillna(0) == 0)
        df.loc[condition_ppm, "Total Sell"] = df.loc[condition_ppm, "Job Ref 1"]

    # 4. If Job Type == 'QUOTED', set Total Sell = Total Sell - Material Sell
    if {"Job Type", "Total Sell", "Material Sell"}.issubset(df.columns):
        condition_quoted = df["Job Type"].str.strip().str.upper() == "QUOTED"
        df.loc[condition_quoted, "Total Sell"] = (
            df.loc[condition_quoted, "Total Sell"].fillna(0)
            - df.loc[condition_quoted, "Material Sell"].fillna(0)
        )

    # 5. Drop Job Ref 1 now (we’ve used it)
    if "Job Ref 1" in df.columns:
        df = df.drop(columns=["Job Ref 1"], errors="ignore")

    # 6. Add Labour column (always = Total Sell - Material Sell)
    if {"Total Sell", "Material Sell"}.issubset(df.columns):
        df["Labour"] = df["Total Sell"].fillna(0) - df["Material Sell"].fillna(0)
    else:
        df["Labour"] = pd.NA

    # 7. Sort by Engineer (A–Z)
    if {"Engineer", "Job Travel"}.issubset(df.columns):
        df = df.sort_values(
            by=["Engineer", "Job Travel"],
            ascending=[True, True]
        ).reset_index(drop=True)
    elif "Engineer" in df.columns:
        df = df.sort_values(by="Engineer", ascending=True).reset_index(drop=True)

    desired_order = [
        "Job Number",
        "Quote Number",
        "Customer Order Number",
        "Job Type",
        "Status",
        "Job Category",
        "Customer Name",
        "Site Name",
        "Engineer",
        "Job Travel",
        "Time on Site",
        "Time off Site",
        "Home Time",
        "Material Cost",
        "Material Sell",
        "Labour",
        "Total Sell"
    ]

    df = df[[c for c in desired_order if c in df.columns] + [c for c in df.columns if c not in desired_order]]

    return df


def connect_ftp() -> FTP_TLS:
    if not FTP_USER or not FTP_PASS:
        raise RuntimeError("FTP_USER and FTP_PASS must be set")

    ftps = FTP_TLS(FTP_HOST)
    ftps.login(FTP_USER, FTP_PASS)
    ftps.prot_p()
    return ftps


def ensure_dir(ftps: FTP_TLS, path: str) -> None:
    """Ensure directory path exists"""
    original = ftps.pwd()
    parts = [p for p in path.strip("/").split("/") if p]

    for part in parts:
        try:
            ftps.cwd(part)
        except error_perm:
            ftps.mkd(part)
            ftps.cwd(part)

    ftps.cwd(original)


def list_csv_files(ftps: FTP_TLS, path: str) -> list[str]:
    """Return list of .csv files"""
    ftps.cwd(path)
    try:
        names = ftps.nlst()
    except error_perm as e:
        if "No files found" in str(e):
            return []
        raise

    return [n for n in names if n.lower().endswith(".csv")]


def file_exists(ftps: FTP_TLS, directory: str, filename: str) -> bool:
    """Check if file exists in directory"""
    ftps.cwd(directory)
    try:
        names = ftps.nlst()
    except error_perm as e:
        if "No files found" in str(e):
            return False
        raise
    return filename in names


def download_csv_to_dataframe(ftps: FTP_TLS, directory: str, filename: str) -> pd.DataFrame:
    """Download csv from ftp into pandas DataFrame"""
    ftps.cwd(directory)
    buf = io.BytesIO()
    ftps.retrbinary(f"RETR {filename}", buf.write)
    buf.seek(0)
    return pd.read_csv(buf)


def upload_dataframe_as_csv(ftps: FTP_TLS, directory: str, filename: str, df: pd.DataFrame) -> None:
    """Upload pandas DataFrame as csv to ftp"""
    ftps.cwd(directory)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    buf = io.BytesIO(csv_bytes)
    ftps.storbinary(f"STOR {filename}", buf)


def delete_file(ftps: FTP_TLS, directory: str, filename: str) -> None:
    try:
        ftps.cwd(directory)
        ftps.delete(filename)
        print(f"Deleted original file: {filename}")
    except Exception as e:
        print(f"Warning: Could not delete {filename}: {e}")


def process_new_files():
    print("Connecting to FTP...")
    ftps = connect_ftp()

    try:
        print(f"Ensuring output directory exists: {OUTPUT_DIR}")
        ensure_dir(ftps, OUTPUT_DIR)

        print(f"Listing CSV files in {INPUT_DIR}...")
        input_files = list_csv_files(ftps, INPUT_DIR)

        if not input_files:
            print("No CSV files found. Nothing to do.")
            return

        for name in input_files:
            base, ext = os.path.splitext(name)
            processed_name = f"{base}_clean.csv"

            if file_exists(ftps, OUTPUT_DIR, processed_name):
                print(f"Skipping {name} (already processed as {processed_name})")
                continue

            print(f"Processing {name} -> {processed_name}")

            df_raw = download_csv_to_dataframe(ftps, INPUT_DIR, name)
            df_clean = transform_dataframe(df_raw)
            upload_dataframe_as_csv(ftps, OUTPUT_DIR, processed_name, df_clean)
            print(f"Uploaded cleaned file to {OUTPUT_DIR}/{processed_name}")
            delete_file(ftps, INPUT_DIR, name)
            

        print("Done processing files.")
    finally:
        try:
            ftps.quit()
            print("FTP connection closed.")
        except Exception as e:
            print(f"Warning: error while closing FTP connection: {e}")


if __name__ == "__main__":
    process_new_files()

  
      
