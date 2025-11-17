import io
import os
from ftplib import FTP_TLS, error_perm
import pandas as pd
import numpy as np

from upload_to_drive import upload_to_drive

FTP_HOST = "ftp.drivehq.com"
FTP_USER = os.environ.get("FTP_USER")
FTP_PASS = os.environ.get("FTP_PASS")

INPUT_DIR = "/JoblogicReports"
OUTPUT_DIR = "/JoblogicReports/processed"

ENGINEER_RATE_WEEKDAY = {
    "Adrian Lewis": 15,
    "Airon Paul": 12,
    "Arron Barnes": 12.50,
    "Bernard Bezuidenhout": 16.50,
    "Bradley Greener-Simon": 16.50,
    "Charlie Rowley": 16.00,
    "Chris Eland": 0,
    "David Head": 0,
    "Ellis Russell": 0,
    "Fabio Conceiocoa": 20,
    "Gary Brunton": 19,
    "Gavain Brown ": 20,
    "Greg Czubak": 0,
    "Jair Gomes": 15,
    "Jake LeBeau": 13,
    "Jamie Scott": 0,
    "Jordan Utter": 15,
    "Kevin Aubignac": 0,
    "Matt Bowden ": 14,
    "Mike Weare": 0,
    "Nelson Vieira": 20,
    "Paul Preston": 15,
    "Richard Lambert": 14.5,
    "Sam Eade": 14,
    "Sharick Bartley": 15,
    "Tom Greener-Simon": 15,
    "William Mcmillan ": 18,
    "Younas": 15,
    "kieran Mbala": 14,
}

ENGINEER_RATE_WEEKEND = {
    "Adrian Lewis": 35,
    "Airon Paul": 35,
    "Arron Barnes": 25,
    "Bernard Bezuidenhout": 35,
    "Bradley Greener-Simon": 35,
    "Charlie Rowley": 35,
    "Chris Eland": 0,
    "David Head": 0,
    "Ellis Russell": 0,
    "Fabio Conceiocoa": 35,
    "Gary Brunton": 35,
    "Gavain Brown": 35,
    "Greg Czubak": 0,
    "Jair Gomes": 35,
    "Jake LeBeau": 25,
    "Jamie Scott": 0,
    "Jordan Utter": 35,
    "Kevin Aubignac": 0,
    "Matt Bowden": 35,
    "Mike Weare": 0,
    "Nelson Vieira": 35,
    "Paul Preston": 35,
    "Richard Lambert": 35,
    "Sam Eade": 35,
    "Sharick Bartley": 35,
    "Tom Greener-Simon": 35,
    "William Mcmillan": 35,
    "Younas": 35,
    "kieran Mbala": 35,
}


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
    for col in ["Total Sell", "Material Sell", "Job Ref 1", "Material Cost"]:
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

    #2c missing home time when there is long break of 4h
    if {"Engineer", "Job Travel", "Time off Site", "Home Time"}.issubset(df.columns):
        df = df.sort_values(by=["Engineer", "Job Travel"]).reset_index(drop=True)

        df["next_travel"] = df.groupby("Engineer")["Job Travel"].shift(-1)

        mask = (
            df["Home Time"].isna()
            & df["Time off Site"].notna()
            & df["next_travel"].notna()
        )

        gap = df["next_travel"] - df["Time off Site"]

        end_of_day = gap.dt.total_seconds() > 4 * 3600
        mask = mask & end_of_day

        df.loc[mask, "Home Time"] = df.loc[mask, "Time off Site"]

        df = df.drop(columns=["next_travel"])

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

    #8 calculate day cost
    if {"Engineer", "Job Travel", "Home Time", "Material Cost", "Material Sell"}.issubset(df.columns):
        df = df.sort_values(by=["Engineer", "Job Travel"]).reset_index(drop=True)

        def add_shift_ids(group: pd.DataFrame) -> pd.DataFrame:
            ht = group["Home Time"]
            is_start = ht.shift(1).notna()
            is_start.iloc[0] = True

            shift_num = is_start.cumsum().astype(int)
            base = str(group["Engineer"].iloc[0])
            group["Shift ID"] = base + "_" + shift_num.astype(str)
            return group

        df = df.groupby("Engineer", group_keys=False).apply(add_shift_ids)

        if {"Time on Site", "Time off Site"}.issubset(df.columns):
            duration = df["Time off Site"] - df["Time on Site"]
            df["_job_hours"] = (duration.dt.total_seconds() / 3600).fillna(0)
        else:
            df["_job_hours"] = 0.0

        #----------------------------------------------------------------
        if {"Job Number", "Engineer", "Time on Site", "Time off Site", "Material Cost", "Material Sell", "Labour", "Total Sell"}.issubset(df.columns):
            # 1️⃣ Calculate worked hours per row
            df["_job_hours_split"] = (
                (df["Time off Site"] - df["Time on Site"])
                .dt.total_seconds() / 3600
            ).fillna(0)

            # 2️⃣ Total hours per job (all engineers combined)
            job_total_hours = df.groupby("Job Number")["_job_hours_split"].transform("sum")

            # 3️⃣ Total hours per engineer per job (sum of all their visits)
            eng_job_hours = df.groupby(["Job Number", "Engineer"])["_job_hours_split"].transform("sum")

            # 4️⃣ Engineer's proportional share (same for all their rows)
            engineer_share = np.where(job_total_hours > 0, eng_job_hours / job_total_hours, 0.0)

            # 5️⃣ Apply engineer share equally across all their rows (not divided by visits)
            for col in ["Material Cost", "Material Sell", "Labour", "Total Sell"]:
                if col in df.columns:
                    df[col] = (df[col].fillna(0) * engineer_share).round(2)

            # 6️⃣ Clean up helper column
            df = df.drop(columns=["_job_hours_split"])
        
        #----------------------------------------------------------------

        shift_totals = (
            df.groupby("Shift ID")
            .agg({
                "Material Cost": "sum",
                "Material Sell": "sum",
                "Labour": "sum",
                "_job_hours": "sum",
                "Job Travel": "min",
                "Home Time": "max",
                "Time off Site": "max",
                "Engineer": "first",
            })
            .rename(columns={
                "Material Cost": "Day Cost",
                "Material Sell": "Day Sell",
                "Labour": "Day Labour",
                "_job_hours": "Day Hours",
                "Job Travel": "Shift Start",
                "Home Time": "Shift End",
                "Time off Site": "Last Time off Site",
            })
        )

        shift_totals["Real Date"] = shift_totals["Shift Start"].dt.date
        shift_totals["Day Part Profit"] = shift_totals["Day Sell"] - shift_totals["Day Cost"]

        OVERHEAD_VALUE = 471.03
        
        shift_totals["Overhead"] = np.where(
            shift_totals["Day Hours"] > 0,
            OVERHEAD_VALUE / shift_totals["Day Hours"],
            0.0,
        ).round(2)

        SPECIAL_ENGS = {"Greg Czubak", "Mike Weare"}
        special_shift_mask = shift_totals["Engineer"].astype(str).str.strip().isin(SPECIAL_ENGS)

        shift_totals.loc[special_shift_mask, "Overhead"] = np.where(
            shift_totals.loc[special_shift_mask, "Day Hours"] > 0,
            600.0 / shift_totals.loc[special_shift_mask, "Day Hours"],
            0.0,
        ).round(2)
            
        # ---------------- WAGE CALCULATION ---------------------

        is_weekend = shift_totals["Shift Start"].dt.weekday >= 5
        weekday_rate = shift_totals["Engineer"].map(ENGINEER_RATE_WEEKDAY)
        weekend_rate = shift_totals["Engineer"].map(ENGINEER_RATE_WEEKEND)

        hourly_rate = weekday_rate.copy()
        hourly_rate[is_weekend & weekend_rate.notna()] = weekend_rate[is_weekend & weekend_rate.notna()]
        hourly_rate = hourly_rate.fillna(0)

        total_duration = (
            (shift_totals["Shift End"] - shift_totals["Shift Start"])
            .dt.total_seconds() / 3600
        ).fillna(0).clip(lower=0)

        # ---- SUBCONTRACTORS (fixed rules) ----
        SUB_CONTRACTORS = {
            "Kevin Aubignac",
            "Ellis Russell",
        }

        # strip spaces from Engineer names before matching
        eng_clean = shift_totals["Engineer"].astype(str).str.strip()
        sc_mask = eng_clean.isin(SUB_CONTRACTORS)
        non_sc_mask = ~sc_mask

        # initialise columns
        shift_totals["Day Basic Wage"] = 0.0
        shift_totals["Day Overtime Wage"] = 0.0
        shift_totals["Total Pay"] = 0.0
        shift_totals["Wage/Pension/NI"] = 0.0

        # --- Subcontractor pay: £90 first hour, £60/hr after, 15-min increments ---
        if sc_mask.any():
            sc_hours = shift_totals.loc[sc_mask, "Day Hours"].fillna(0)

            has_hours = (sc_hours > 0).astype(int)
            first_hour_charge = 90 * has_hours

            extra_hours = (sc_hours - 1).clip(lower=0)
            extra_hours_rounded = (np.ceil(extra_hours / 0.25) * 0.25).round(2)
            extra_charge = extra_hours_rounded * 60

            sc_total_pay = first_hour_charge + extra_charge

            shift_totals.loc[sc_mask, "Day Basic Wage"] = sc_total_pay
            shift_totals.loc[sc_mask, "Day Overtime Wage"] = 0.0
            shift_totals.loc[sc_mask, "Total Pay"] = sc_total_pay
            # No uplift for NI/pension for subcontractors
            shift_totals.loc[sc_mask, "Wage/Pension/NI"] = sc_total_pay

        # ---- Standard employees (existing logic) ----
        home_travel_duration = (
            (shift_totals["Shift End"] - shift_totals["Last Time off Site"])
            .dt.total_seconds() / 3600
        ).fillna(0).clip(lower=0)
            
        pre_home_duration = (
            (shift_totals["Last Time off Site"] - shift_totals["Shift Start"])
            .dt.total_seconds() / 3600
        ).fillna(0).clip(lower=0)

        weekday_mask = non_sc_mask & (~is_weekend)
        weekend_mask = non_sc_mask & is_weekend

        basic_hours = pd.Series(0.0, index=shift_totals.index)
        overtime_hours = pd.Series(0.0, index=shift_totals.index)
        extra_drive = pd.Series(0.0, index=shift_totals.index)

        # ----------------- WEEKDAYS (Mon–Fri) -----------------
        basic_hours.loc[weekday_mask & (total_duration > 0)] = 9.0

        weekday_overtime = (pre_home_duration - 9).clip(lower=0)
        weekday_overtime.loc[total_duration <= 9] = 0.0
        overtime_hours.loc[weekday_mask] = weekday_overtime.loc[weekday_mask]

        weekday_extra_drive = (total_duration - 9).clip(lower=0)
        weekday_extra_drive = weekday_extra_drive.clip(upper=home_travel_duration)
        weekday_extra_drive.loc[total_duration <= 9] = 0.0
        extra_drive.loc[weekday_mask] = weekday_extra_drive.loc[weekday_mask]

        # ----------------- WEEKENDS (Sat–Sun) -----------------
        basic_hours.loc[weekend_mask] = shift_totals.loc[weekend_mask, "Day Hours"].fillna(0)
        overtime_hours.loc[weekend_mask] = 0.0
        extra_drive.loc[weekend_mask] = 0.0

        overtime_factor = pd.Series(1.5, index=shift_totals.index)
        overtime_factor.loc[weekend_mask] = 1.0

        # apply standard wage logic ONLY to non-subcontractors
        shift_totals.loc[non_sc_mask, "Day Basic Wage"] = (
            basic_hours[non_sc_mask] * hourly_rate[non_sc_mask]
        ).round(2)

        shift_totals.loc[non_sc_mask, "Day Overtime Wage"] = (
            overtime_hours[non_sc_mask] * hourly_rate[non_sc_mask] * overtime_factor[non_sc_mask]
            + extra_drive[non_sc_mask] * hourly_rate[non_sc_mask]
        ).round(2)

        shift_totals.loc[non_sc_mask, "Total Pay"] = (
            shift_totals.loc[non_sc_mask, "Day Basic Wage"]
            + shift_totals.loc[non_sc_mask, "Day Overtime Wage"]
        ).round(2)

        shift_totals.loc[non_sc_mask, "Wage/Pension/NI"] = (
            (shift_totals.loc[non_sc_mask, "Day Basic Wage"] * 0.03
             + shift_totals.loc[non_sc_mask, "Total Pay"]) * 1.1435
        ).round(2)

        #---------------------------------------------------------------------------------------

        df = df.join(shift_totals[["Day Cost", "Day Sell", "Day Labour", "Day Hours", "Real Date", "Day Part Profit", "Day Basic Wage", "Day Overtime Wage", "Total Pay", "Wage/Pension/NI", "Overhead",]], on="Shift ID")

        df["Overhead without Wage"] = pd.NA 
        df["Total Cost"] = pd.NA
        summary_idx = df.groupby("Shift ID").tail(1).index
        mask_summary = df.index.isin(summary_idx)

        df.loc[mask_summary, "Overhead without Wage"] = OVERHEAD_VALUE
        special_mask = mask_summary & df["Engineer"].astype(str).str.strip().isin(SPECIAL_ENGS)
        df.loc[special_mask, "Overhead without Wage"] = 600.0

        for col in ["Day Basic Wage", "Day Overtime Wage", "Total Pay", "Wage/Pension/NI"]:
            df.loc[special_mask, col] = 0.0
        
        df.loc[mask_summary, "Total Cost"] = (df.loc[mask_summary, "Wage/Pension/NI"].fillna(0) + df.loc[mask_summary, "Overhead without Wage"].fillna(0)).round(2)
        df.loc[mask_summary, "Labour Profit"] = (df.loc[mask_summary, "Day Labour"].fillna(0) - df.loc[mask_summary, "Total Cost"].fillna(0)).round(2)
        df.loc[mask_summary, "Labour Margin"] = (df.loc[mask_summary, "Labour Profit"] / df.loc[mask_summary, "Day Labour"]).replace([np.inf, -np.inf], np.nan) .fillna(0) * 100
        df.loc[mask_summary, "Labour Margin"] = df.loc[mask_summary, "Labour Margin"].round(2)
        df.loc[mask_summary, "Total Profit"] = (df.loc[mask_summary, "Labour Profit"].fillna(0) + df.loc[mask_summary, "Day Part Profit"].fillna(0)).round(2)

        labour_profit = df.loc[mask_summary, "Labour Profit"].fillna(0)
        parts_profit = df.loc[mask_summary, "Day Part Profit"].fillna(0)
        labour_turnover = df.loc[mask_summary, "Day Labour"].fillna(0)
        part_sell = df.loc[mask_summary, "Day Sell"].fillna(0)

        # Combined Margin = (Labour Profit + Parts Profit) / (Labour Turnover + Parts Sell)
        combined_margin = (labour_profit + parts_profit) / (labour_turnover + part_sell)
        combined_margin = combined_margin.replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        df.loc[mask_summary, "Combined Margin"] = combined_margin.round(2)

        conditions = [
            df.loc[mask_summary, "Combined Margin"] <= 0,
            (df.loc[mask_summary, "Combined Margin"] > 0) & (df.loc[mask_summary, "Combined Margin"] <20),
            (df.loc[mask_summary, "Combined Margin"] >= 20) & (df.loc[mask_summary, "Combined Margin"] < 35),
            (df.loc[mask_summary, "Combined Margin"] >= 35) & (df.loc[mask_summary, "Combined Margin"] < 50),
            df.loc[mask_summary, "Combined Margin"] >=50,
        ]
        choices = [-20, 0, 20, 50, 100]

        df.loc[mask_summary, "Bonus"] = np.select(conditions, choices, default=0)

        for col in ["Day Cost", "Day Sell", "Day Labour", "Day Hours", "Real Date","Day Part Profit", "Day Basic Wage", "Day Overtime Wage", "Overhead without Wage", "Total Cost", "Total Pay", "Wage/Pension/NI",]:
            df.loc[~mask_summary, col] = pd.NA
            
        df = df.drop(columns=["Shift ID", "_job_hours"])
    else:
        df["Day Cost"] = pd.NA
        df["Day Sell"] = pd.NA
        df["Day Labour"] = pd.NA
        df["Day Hours"] = pd.NA
        df["Real Date"] = pd.NA
        df["Day Part Profit"] = pd.NA
        df["Day Basic Wage"] = pd.NA
        df["Day Overtime Wage"] = pd.NA
        df["Total Pay"] = pd.NA
        df["Wage/Pension/NI"] = pd.NA
        df["Overhead without Wage"] = pd.NA
        df["Total Cost"] = pd.NA
        

    #9 makes sure these columns exsit
    for col in ["Overhead", "Day Cost", "Day Sell", "Day Labour", "Day Hours", "Real Date", "Day Part Profit", "Day Basic Wage", "Day Overtime Wage", "Total Pay", "Wage/Pension/NI", "Overhead without Wage", "Total Cost",]:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.rename(columns={
        "Day Cost": "Part Cost",
        "Day Sell": "Part Sell",
        "Day Labour": "Labour Turnover",
        "Overhead": "Overhead Per Job",
        "Overhead without Wage": "Overhead",
        "Day Part Profit": "Parts Profit"
    })

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
        "Real Date",
        "Material Cost",
        "Material Sell",
        "Labour",
        "Total Sell",
        "Day Hours",
        "Overhead Per Job",
        "Day Basic Wage",
        "Day Overtime Wage",
        "Total Pay",
        "Wage/Pension/NI",
        "Overhead",
        "Total Cost",
        "Part Cost",
        "Part Sell",
        "Parts Profit",
        "Labour Turnover",
        "Labour Profit",
        "Labour Margin",
        "Combined Margin",
        "Total Profit",
        "Bonus",
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

            local_filename = processed_name
            local_path = os.path.join("/tmp", local_filename)
            df_clean.to_csv(local_path, index=False)
            print(f"Saved local file for drive upload: {local_path}")

            upload_to_drive(local_path, drive_filename=local_filename)
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
  
