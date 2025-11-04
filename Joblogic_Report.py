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

  #1. Drops columns
  columns_to_drop = ["Job Ref 2","Expense Cost","Expense Sell","Reference Number"]
  df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

  #2. Convert Numeric columns 
  for col in ["Total Sell", "Material Sell", "Job Ref 1"]:
    if col in df.columns:
      df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True),
        errors="coerce",
      )

  #3 add Labour column (Total
  
  if "Total Sell" in df.columns and "Material Sell" in df.columns:
    total = pd.to_numeric(
      df["Total Sell"].astype(str).str.replace("[^0-9\.\-]", "", regex=True),
      errors="coerce",
    )
    material = pd.to_numeric(
      df["Material Sell"].astype(str).str.replace(r"[0-9]\.\-]", "", regex=True),
      errors="coerce",
    )
    df["Labour"] = total - material
  else:
    df["Labour"] = pd.NA

  if {"Job Type", "Total Sell", "Job Ref 1"}.issubset(df.columns):
    df["Total Sell"] = pd.to_numeric(
      df["Total Sell"].astype(str).str.replace(r"0-9\.\-]", "", regex=True),
      errors="coerce",
    )
    df["Job Ref 1"] = pd.to_numeric(
      df["Job Ref 1"].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True),
      errors="coerce",
    )

  condition = (df["Job Type"].str.strip().str.upper() == "PPM") & (df["Total Sell"].fillna(0) == 0)
  df.loc[condition, "Total Sell"] = df.loc[condition, "Job Ref 1"]

  df = df.drop(columns=["Job Ref 1"], errors="ignore")
  
  return df
  
def connect_ftp() -> FTP_TLS:
  if not FTP_USER or not FTP_PASS:
    raise RuntimeError("FTP USER AND PASS MUST BE SET")

  ftps = FTP_TLS(FTP_HOST)
  ftps.login(FTP_USER, FTP_PASS)
  ftps.prot_p()
  return ftps

def ensure_dir(ftps: FTP_TLS, path: str) -> None:
  "Ensure directory path exists"
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
  "Return list of .csv files"
  ftps.cwd(path)
  try:
    names = ftps.nlst()
  except error_perm as e:
    if "No files found" in str(e):
      return []
    raise

  return [n for n in names if n.lower().endswith(".csv")]

def file_exists(ftps: FTP_TLS, directory: str, filename: str) -> bool:
  "Check if file exsits in directory"

    ftps.cwd(directory)
    try:
      names = ftps.nlst()
    expect error_perm as e:
      if "No files found" in str(e):
        return False
      raise
    return filename in names

def download_csv_to_dataframe(ftps: FTP_TLS, directory: str, filename: str) -> pd.DataFrame:
  "Download csv from ftp into panda data frame"
  ftps.cwd(directory)
  buf = io.BytesIO()
  ftps.retrbinary(f"RETR {filename}", buf.write)
  buf.seek(0)
  return pd.read_csv(buf)

def upload_dataframe_as_csv(ftps: FTP_TLS, directory: str, filename: str, df: pd.DataFrame) -> None
  "Upload pandas DataFrame as csv to ftp"

  ftps.cwd(directory)
  csv_bytes = df.to_csv(index=False).encode("utf-8")
  buf = io.BytesIO(csv_bytes)
  ftps.storbinary(f"STOR {filename}", buf)

def delete_file(ftps: FTP_TLS, directory: str, filename: str) -> None;
  ftps.cwd(directory)
  ftps.delete(filename)

#---------------------------------------------
def process_new_files():
  print(f"connecting to ftp...")
  ftps = connect_ftp()

  try:
    print(f"Ensuring output directory exsits: {OUTPUT_DIR}")
    ensure_dir(ftps, OUTPUT_DIR)

    print(f"ListingCSV files in {INPUT_DIR}...")
    input_files = list_csv_files(ftps, INPUT_DIR)

    if not input_files:
      print("No CSV files found. Nothing to do.")
      return

    for name in input_files:
      base, ext = os.path.splitext(name)
      processed_name = f"{base}_clean.csv"

      if file_exsits(ftps, OUTPUT_DIR, processed_name):
        print(f"Skipping {name} (already processed as {processed_name})")
        continue

      print(f"Processing {name} -> {processed_name}")

      df_raw = download_csv_to_dataframe(ftps, INPUT_DIR, name)
      df_clean = transform_dataframe(df_raw)
      upload_dataframe _as_csv(ftps, OUTPUT_DIR, processed_name, df_clean)
      print(f"Uploaded cleaned file to {OUTPUT_DIR}/{processed_name}")

    print("Done processing files.")
  finally:
    ftps.quit()
    print("FTP connection closed.")

if __name__ == "__main__":
  process_new_files()
  
      
