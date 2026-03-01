import io
import os
from ftplib import FTP_TLS, error_perm
import pandas as pd
import numpy as np

from upload_to_drive import upload_to_drive
from datetime import date, timedelta, datetime

# ==============================================================================
# FTP CONFIG
# ==============================================================================
FTP_HOST = "ronnie123.synology.me"
FTP_USER = os.environ.get("FTP_USER")
FTP_PASS = os.environ.get("FTP_PASS")

INPUT_DIR  = "/JoblogicFTP"
OUTPUT_DIR = "/JoblogicFTP/processed"

# ==============================================================================
# ENGINEER CONFIGURATION
# All engineer lists and rates live here — edit this section to add/update people
# ==============================================================================

# Weekday hourly rates (0 = not directly paid / assistant / rate handled elsewhere)
ENGINEER_RATE_WEEKDAY = {
    "Adrian Lewis": 0,
    "Airon Paul": 0,
    "Arron Barnes": 0,
    "Bernard Bezuidenhout": 15,
    "Bradley Greener-Simon": 15.00,
    "Charlie Rowley": 16.00,
    "Chris Eland": 0,
    "Ellis Russell": 0,
    "Fabio Conceiocoa": 17.50,
    "Gary Brunton": 17.00,
    "Gavain Brown ": 17.50,
    "Greg Czubak": 0,
    "Jair Gomes": 0,
    "Jake LeBeau": 0,
    "Jamie Boyd": 25,
    "Jamie Scott": 0,
    "Jordan Utter": 0,
    "Kevin Aubignac": 0,
    "Matt Bowden ": 14,
    "Mike Weare": 0,
    "Nelson Vieira": 17.50,
    "Paul Preston": 15,
    "Richard Lambert": 14.5,
    "Sam Eade": 0,
    "Sharick Bartley": 0,
    "Tom Greener-Simon": 0,
    "William Mcmillan ": 18,
    "Younas": 0,
    "kieran Mbala": 0,
    "Iosua Caloro": 0,
    "Stefan Caloro": 0,
    "Oskars Perkons": 0,
    "Mikael Williams": 0,
    "Jack Morbin": 0,
    "Alfie Pateman": 0,
    "Jaydan Brown": 0,
    "Bartosz Skalbania": 20,
    "Aidan KIngsbury Cleghorn": 0,
    "Zain Saeed": 0,
}

# Weekend hourly rates
ENGINEER_RATE_WEEKEND = {
    "Adrian Lewis": 0,
    "Airon Paul": 0,
    "Arron Barnes": 0,
    "Bernard Bezuidenhout": 35,
    "Bradley Greener-Simon": 35,
    "Charlie Rowley": 35,
    "Chris Eland": 0,
    "Ellis Russell": 0,
    "Fabio Conceiocoa": 35,
    "Gary Brunton": 35,
    "Gavain Brown ": 35,
    "Greg Czubak": 0,
    "Jair Gomes": 0,
    "Jake LeBeau": 0,
    "Jamie Boyd": 25,
    "Jamie Scott": 0,
    "Jordan Utter": 0,
    "Kevin Aubignac": 0,
    "Matt Bowden": 35,
    "Mike Weare": 0,
    "Nelson Vieira": 35,
    "Paul Preston": 35,
    "Richard Lambert": 35,
    "Sam Eade": 0,
    "Sharick Bartley": 0,
    "Tom Greener-Simon": 0,
    "William Mcmillan ": 35,
    "Younas": 0,
    "kieran Mbala": 0,
    "Iosua Caloro": 0,
    "Stefan Caloro": 0,
    "Oskars Perkons": 0,
    "Mikael Williams": 0,
    "Jack Morbin": 0,
    "Alfie Pateman": 0,
    "Jaydan Brown": 35,
    "Bartosz Skalbania": 35,
    "Aidan KIngsbury Cleghorn": 0,
    "Zain Saeed": 0,
}

# Pay rises: name -> list of (effective_date, new_weekday_rate, new_weekend_rate)
# Multiple entries per person are applied oldest-first so the latest rise always wins
RATE_CHANGES = {
    "Bernard Bezuidenhout": [(date(2025, 6, 24), 16.50, 35.00)],
    "kieran Mbala":         [(date(2025, 6,  3), 14.00, 35.00)],
    "Sam Eade":             [(date(2024, 1,  4), 12.50, 35.00),
                             (date(2025, 8, 26), 14.00, 35.00)],
    "Gavain Brown ":        [(date(2025, 6, 24), 20.00, 35.00)],
    "Nelson Vieira":        [(date(2025, 6, 24), 20.00, 35.00)],
    "Gary Brunton":         [(date(2024, 9, 24), 19.00, 35.00)],
    "Fabio Conceiocoa":     [(date(2025, 6, 24), 20.00, 35.00)],
    "Bradley Greener-Simon":[(date(2025, 5, 27), 16.50, 35.00)],
    "Sharick Bartley":      [(date(2025, 4,  8), 15.00, 35.00),
                             (date(2025,12, 30), 17.00, 35.00)],
    "Younas":               [(date(2025, 4, 22), 15.00, 35.00),
                             (date(2025,12, 30), 17.00, 35.00)],
    "Tom Greener-Simon":    [(date(2025, 8, 26), 15.00, 35.00)],
    "Adrian Lewis":         [(date(2024, 8, 27), 15.00, 35.00)],
    "Airon Paul":           [(date(2025,12, 10), 15.00, 35.00)],
}

# Assistant -> Engineer promotion dates (Assistant BEFORE date, Engineer ON/AFTER date)
# Adding a new person here automatically handles their Role column — no other code needed
ASSISTANT_CUTOFFS = {
    "Airon Paul":        date(2025, 12, 10),
    "kieran Mbala":      date(2025,  6,  3),
    "Sam Eade":          date(2024,  1,  4),
    "Sharick Bartley":   date(2025,  4,  8),
    "Younas":            date(2025,  4, 22),
    "Tom Greener-Simon": date(2025,  8, 26),
    "Adrian Lewis":      date(2024,  8, 27),
}

# People who get zero job revenue share (like assistants for cost purposes)
# Jamie Boyd is here so he takes no share of job labour/materials
# NOTE: Jamie Boyd is NOT in ASSISTANTS_FOR_ROLE so his Role column still shows "Engineer"
ASSISTANTS = {
    "Airon Paul",
    "Arron Barnes",
    "Iosua Caloro",
    "Stefan Caloro",
    "Diogo Barroso",
    "Jair Gomes",
    "Jake LeBeau",
    "Jamie Boyd",
    "Jamie Scott",
    "Jordan Utter",
    "Oskars Perkons",
    "Mikael Williams",
    "Jack Morbin",
    "kieran Mbala",
    "Sharick Bartley",
    "Younas",
    "Tom Greener-Simon",
    "Adrian Lewis",
    "Zain Saeed",
}

# Assistants for Role column display only (subset of ASSISTANTS — excludes Jamie Boyd)
ASSISTANTS_FOR_ROLE = ASSISTANTS - {"Jamie Boyd"}

# Subcontractors: special pay rules (£90 first hour, £60/hr after in 15-min increments)
SUB_CONTRACTORS = {
    "Kevin Aubignac",
    "Ellis Russell",
}

# Subcontractors for Role column display
SUBCONTRACTORS_FOR_ROLE = SUB_CONTRACTORS | {"Greg Czubak", "Mike Weare"}

# Engineers who get 10-hour basic days instead of the standard 9
TEN_HOUR_ENGINEERS = {"Paul Preston"}

# Engineers paid on-site hours only: flat rate, 7-hour minimum retainer, no overtime
# multiplier, no job revenue share (see ASSISTANTS above)
ON_SITE_PAY_ENGINEERS  = {"Jamie Boyd"}
ON_SITE_RETAINER_HOURS = 7

# Weekend minimum retainer (hours) for standard engineers
WEEKEND_RETAINER_HOURS = 5

# Engineers with special (higher) overhead value
SPECIAL_ENGS          = {"Greg Czubak", "Mike Weare"}
SPECIAL_OVERHEAD_VALUE = 600.0

# Engineers with zero overhead
ZERO_OVERHEAD_ENGS = {"Chris Eland"}

# Management doing jobs: zero wage, zero overhead, zero bonus, Role = "Office"
MANAGEMENT_ENGINEERS = {
    "Chris Rivoire",
    "Chris Smith",
    "David Head",
    "Edward Dale Cooke",
    "Ellie Mahoney",
    "Kathryn Barnes",
    "Steve Elder",
}

# Standard overhead value applied per on-site hour
OVERHEAD_VALUE = 471.03

# Night shift workers (used for Shift Type column)
NIGHT_WORKERS = {
    "Adrian Lewis",
    "Airon Paul",
    "Bernard Bezuidenhout",
    "Diogo Barroso",
    "Fabio Conceiocoa",
    "Gavain Brown",
    "Jack Morbin",
    "Jair Gomes",
    "Jamie Scott",
    "Jordan Utter",
    "Mike Weare",
    "Nelson Vieira",
    "Sharick Bartley",
    "Younas",
    "Zain Saeed",
}

# ==============================================================================
# PAY MONTH CALCULATION
# Pay month runs from last Tuesday of previous calendar month
# through to the last Monday (with Tue+Wed still in month) of current month
# ==============================================================================
def compute_pay_month(day) -> str:
    """Return the pay month label (YYYY-MM) for a given date."""
    if pd.isna(day):
        return pd.NA

    d = pd.to_datetime(day).date()
    y, m = d.year, d.month

    # --- end of pay month: last Monday in month with Tue+Wed still in month ---
    first_next = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    last_day = first_next - timedelta(days=1)

    cur = last_day - timedelta(days=2)
    end = None
    while cur.month == m:
        if cur.weekday() == 0:  # Monday
            end = cur
            break
        cur -= timedelta(days=1)
    if end is None:
        cur = last_day
        while cur.month == m and cur.weekday() != 0:
            cur -= timedelta(days=1)
        end = cur

    # --- previous month ---
    py, pm = (y - 1, 12) if m == 1 else (y, m - 1)
    prev_first_next = date(py + 1, 1, 1) if pm == 12 else date(py, pm + 1, 1)
    prev_last = prev_first_next - timedelta(days=1)

    # start = last Tuesday in previous month with a Wednesday still in that month
    cur2 = prev_last - timedelta(days=1)
    start = None
    while cur2.month == pm:
        if cur2.weekday() == 1 and (cur2 + timedelta(days=1)).month == pm:
            start = cur2
            break
        cur2 -= timedelta(days=1)
    if start is None:
        cur2 = prev_last
        while cur2.month == pm and cur2.weekday() != 1:
            cur2 -= timedelta(days=1)
        start = cur2

    # --- assign to pay month ---
    if d < start:
        yy, mm = (py - 1, 12) if pm == 1 else (py, pm - 1)
    elif d > end:
        yy, mm = (y + 1, 1) if m == 12 else (y, m + 1)
    else:
        yy, mm = y, m

    return f"{yy}-{mm:02d}"


# ==============================================================================
# HELPER: resolve row_date column
# ==============================================================================
def _get_row_date(df: pd.DataFrame) -> pd.Series:
    """Return a date Series using Real Date (Each Row) if present, else Job Travel."""
    if "Real Date (Each Row)" in df.columns:
        return pd.to_datetime(df["Real Date (Each Row)"], errors="coerce").dt.date
    return pd.to_datetime(df["Job Travel"], errors="coerce").dt.date


# ==============================================================================
# PARTS FILE TRANSFORM
# ==============================================================================
def transform_parts_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(1).astype(int)
        df.loc[df["Quantity"] < 1, "Quantity"] = 1
        df = df.loc[df.index.repeat(df["Quantity"])].reset_index(drop=True)

    rename_map = {
        "JobID":           "JobID",
        "VisitStartDate":  "Visit Start",
        "VisitEndDate":    "Visit End",
        "Site":            "Site",
        "PartNumber":      "Part Number",
        "PartDescription": "Part Description",
        "Engineer":        "Engineer",
        "Cost":            "Part Cost",
        "Sell":            "Part Sell",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    keep = ["JobID", "Visit Start", "Visit End", "Site",
            "Part Number", "Part Description", "Engineer", "Part Cost", "Part Sell"]
    df = df[[c for c in keep if c in df.columns]].copy()

    for col in ["JobId", "Site", "Part Number", "Part Description", "Engineer"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in ["Visit Start", "Visit End"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    for col in ["Part Cost", "Part Sell"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True),
                errors="coerce",
            ).fillna(0).round(2)

    if "JobId" in df.columns:
        df = df[df["JobId"].notna() & (df["JobId"].str.len() > 0)].copy()

    if "Visit Start" in df.columns:
        df = df.sort_values(by="Visit Start", ascending=True).reset_index(drop=True)

    return df


# ==============================================================================
# MAIN JOBS FILE TRANSFORM
# ==============================================================================
def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up the Joblogic export:
    - Remove cancelled jobs and future rows
    - PPM: if Total Sell == 0, use Job Ref 1 as Total Sell
    - Quoted (Q0 prefix): Total Sell = Total Sell - Material Sell
    - Labour = Total Sell - Material Sell
    - Calculate shift totals, wages, overhead, profit, margins and bonuses
    """

    # --- Remove cancelled ---
    if "Status" in df.columns:
        df = df.loc[df["Status"].astype(str).str.strip().str.upper() != "CANCELLED"].copy()

    # --- Drop unwanted columns ---
    df = df.drop(columns=[
        "Job Ref 2", "Expense Cost", "Expense Sell",
        "Reference Number", "Quoted Number", "Completed Date",
    ], errors="ignore")

    # --- Convert numeric columns ---
    for col in ["Total Sell", "Material Sell", "Job Ref 1", "Material Cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True),
                errors="coerce",
            )

    # --- Convert datetime columns ---
    for col in ["Job Travel", "Time on Site", "Time off Site", "Home Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), dayfirst=True, errors="coerce")

    # If Job Travel missing but Time on Site present, use Time on Site
    if {"Job Travel", "Time on Site"}.issubset(df.columns):
        mask = df["Job Travel"].isna() & df["Time on Site"].notna()
        df.loc[mask, "Job Travel"] = df.loc[mask, "Time on Site"]

    # --- Remove future rows ---
    today = date.today()
    date_cols = [c for c in ["Job Travel", "Time on Site", "Time off Site", "Home Time"] if c in df.columns]
    if date_cols:
        row_max = df[date_cols].max(axis=1)
        df = df.loc[row_max.isna() | (row_max.dt.date <= today)].copy()

    # --- Sort ---
    if {"Engineer", "Job Travel"}.issubset(df.columns):
        df = df.sort_values(by=["Engineer", "Job Travel"], ascending=[True, True]).reset_index(drop=True)
    elif "Engineer" in df.columns:
        df = df.sort_values(by="Engineer", ascending=True).reset_index(drop=True)

    # --- Fill missing Home Time when gap to next job is > 8 hours ---
    if {"Engineer", "Job Travel", "Time off Site", "Home Time"}.issubset(df.columns):
        df = df.sort_values(by=["Engineer", "Job Travel"]).reset_index(drop=True)
        df["next_travel"] = df.groupby("Engineer")["Job Travel"].shift(-1)
        mask = (
            df["Home Time"].isna()
            & df["Time off Site"].notna()
            & df["next_travel"].notna()
            & ((df["next_travel"] - df["Time off Site"]).dt.total_seconds() > 8 * 3600)
        )
        df.loc[mask, "Home Time"] = df.loc[mask, "Time off Site"]
        df = df.drop(columns=["next_travel"])

    # --- PPM: use Job Ref 1 as Total Sell when Total Sell is 0 ---
    if {"Job Type", "Total Sell", "Job Ref 1"}.issubset(df.columns):
        condition_ppm = (
            df["Job Type"].str.strip().str.upper() == "PPM"
        ) & (df["Total Sell"].fillna(0) == 0)
        df.loc[condition_ppm, "Total Sell"] = df.loc[condition_ppm, "Job Ref 1"]

    # --- Quoted (Q0 prefix only): Total Sell = Total Sell - Material Sell ---
    if {"Job Type", "Total Sell", "Material Sell", "Job Number"}.issubset(df.columns):
        mask_fix = (
            df["Job Type"].astype(str).str.strip().str.upper().eq("QUOTED")
            & df["Job Number"].astype(str).str.strip().str.upper().str.startswith("Q0")
        )
        df.loc[mask_fix, "Total Sell"] = (
            df.loc[mask_fix, "Total Sell"].fillna(0)
            - df.loc[mask_fix, "Material Sell"].fillna(0)
        )

    df = df.drop(columns=["Job Ref 1"], errors="ignore")

    # --- Labour = Total Sell - Material Sell ---
    if {"Total Sell", "Material Sell"}.issubset(df.columns):
        df["Labour"] = df["Total Sell"].fillna(0) - df["Material Sell"].fillna(0)
    else:
        df["Labour"] = pd.NA

    # ==========================================================================
    # SHIFT & WAGE CALCULATIONS
    # ==========================================================================
    if not {"Engineer", "Job Travel", "Home Time", "Material Cost", "Material Sell"}.issubset(df.columns):
        for col in ["Day Cost", "Day Sell", "Day Labour", "Day Hours", "Real Date",
                    "Day Part Profit", "Day Basic Wage", "Day Overtime Wage", "Total Pay",
                    "Wage/Pension/NI", "Overhead without Wage", "Total Cost", "Shift Hours"]:
            df[col] = pd.NA
    else:
        df = df.sort_values(by=["Engineer", "Job Travel"]).reset_index(drop=True)

        # --- Real Date: handle night shifts (jobs before 07:00 with < 8h gap belong to previous day) ---
        jt = df["Job Travel"]
        if "Time off Site" in df.columns:
            prev_off  = df.groupby("Engineer")["Time off Site"].shift(1)
            gap_hours = (jt - prev_off).dt.total_seconds() / 3600
        else:
            gap_hours = pd.Series(np.nan, index=df.index)

        early      = jt.dt.hour < 7
        long_rest  = gap_hours.isna() | (gap_hours >= 8)
        use_prev   = early & (~long_rest)

        real_date = jt.dt.date.copy()
        real_date[use_prev] = (jt[use_prev] - pd.Timedelta(days=1)).dt.date

        df["Real Date"]          = real_date
        df["Real Date (Each Row)"] = real_date

        df["Shift ID"] = df["Engineer"].astype(str).str.strip() + "_" + real_date.astype(str)

        if {"Time on Site", "Time off Site"}.issubset(df.columns):
            df["_job_hours"] = ((df["Time off Site"] - df["Time on Site"])
                                .dt.total_seconds() / 3600).fillna(0)
        else:
            df["_job_hours"] = 0.0

        # --- Revenue share: split job value proportionally by on-site hours ---
        if {"Job Number", "Engineer", "Time on Site", "Time off Site",
            "Material Cost", "Material Sell", "Labour", "Total Sell"}.issubset(df.columns):

            df["_job_hours_split"] = ((df["Time off Site"] - df["Time on Site"])
                                      .dt.total_seconds() / 3600).fillna(0)

            job_total_hours          = df.groupby("Job Number")["_job_hours_split"].transform("sum")
            eng_job_hours            = df.groupby(["Job Number", "Engineer"])["_job_hours_split"].transform("sum")
            engineer_share           = np.where(job_total_hours > 0, eng_job_hours / job_total_hours, 0.0)
            row_within_engineer_share = np.where(eng_job_hours > 0, df["_job_hours_split"] / eng_job_hours, 0.0)
            row_share = pd.Series(engineer_share * row_within_engineer_share, index=df.index, dtype=float)

            # Assistants (and Jamie Boyd) get zero share when a main engineer is on the job
            eng_clean  = df["Engineer"].astype(str).str.strip()
            row_date   = _get_row_date(df)
            is_asst    = eng_clean.isin(ASSISTANTS)
            for name, cutoff in ASSISTANT_CUTOFFS.items():
                is_asst = is_asst & ~(eng_clean.eq(name) & row_date.notna() & (row_date >= cutoff))

            has_main       = (~is_asst).groupby(df["Job Number"]).transform("any")
            row_share_adj  = row_share.copy()
            row_share_adj[is_asst & has_main] = 0.0
            sum_shares     = row_share_adj.groupby(df["Job Number"]).transform("sum")
            row_share_final = row_share_adj.copy()
            renorm          = has_main & (sum_shares > 0)
            row_share_final[renorm] = row_share_adj[renorm] / sum_shares[renorm]

            for col in ["Material Cost", "Material Sell", "Labour", "Total Sell"]:
                if col in df.columns:
                    df[col] = (df[col].fillna(0) * row_share_final).round(2)

            df = df.drop(columns=["_job_hours_split"])

        # --- Aggregate to shift level ---
        shift_totals = (
            df.groupby("Shift ID")
            .agg({
                "Material Cost": "sum",
                "Material Sell": "sum",
                "Labour":        "sum",
                "_job_hours":    "sum",
                "Job Travel":    "min",
                "Home Time":     "max",
                "Time off Site": "max",
                "Time on Site":  "min",
                "Engineer":      "first",
            })
            .rename(columns={
                "Material Cost": "Day Cost",
                "Material Sell": "Day Sell",
                "Labour":        "Day Labour",
                "_job_hours":    "Day Hours",
                "Job Travel":    "Shift Start",
                "Home Time":     "Shift End",
                "Time off Site": "Last Time off Site",
                "Time on Site":  "First Time on Site",
            })
        )

        shift_totals["Shift First Job Travel"]    = df.groupby("Shift ID")["Job Travel"].transform("min").groupby(df["Shift ID"]).first()
        shift_totals["Shift First Time on Site"]  = df.groupby("Shift ID")["Time on Site"].transform("min").groupby(df["Shift ID"]).first()
        shift_totals["Shift Last Time off Site"]  = df.groupby("Shift ID")["Time off Site"].transform("max").groupby(df["Shift ID"]).first()
        shift_totals["Shift Home Time"]           = df.groupby("Shift ID")["Home Time"].transform("max").groupby(df["Shift ID"]).first()

        shift_totals["First Job to Last Job Hours"] = (
            (shift_totals["Last Time off Site"] - shift_totals["First Time on Site"])
            .dt.total_seconds() / 3600
        ).fillna(0).round(2)

        shift_totals["Day Part Profit"] = shift_totals["Day Sell"] - shift_totals["Day Cost"]
        shift_totals["Real Date"]       = shift_totals["Shift Start"].dt.date
        shift_totals["Pay Month"]       = shift_totals["Real Date"].apply(compute_pay_month)

        # --- Overhead ---
        eng_shift  = shift_totals["Engineer"].astype(str).str.strip()
        shift_date = shift_totals["Shift Start"].dt.date

        shift_totals["Overhead"] = np.where(
            shift_totals["Day Hours"] > 0,
            OVERHEAD_VALUE / shift_totals["Day Hours"],
            0.0,
        ).round(2)

        # Special engineers get higher overhead
        sp_mask = eng_shift.isin(SPECIAL_ENGS)
        shift_totals.loc[sp_mask, "Overhead"] = np.where(
            shift_totals.loc[sp_mask, "Day Hours"] > 0,
            SPECIAL_OVERHEAD_VALUE / shift_totals.loc[sp_mask, "Day Hours"],
            0.0,
        ).round(2)

        # Assistants (and Jamie Boyd) get zero overhead
        asst_shift_mask = eng_shift.isin(ASSISTANTS)
        for name, cutoff in ASSISTANT_CUTOFFS.items():
            m = eng_shift.eq(name) & shift_date.notna()
            asst_shift_mask = asst_shift_mask & ~(m & (shift_date >= cutoff))
        shift_totals.loc[asst_shift_mask, "Overhead"] = 0.0

        # Zero overhead engineers
        shift_totals.loc[eng_shift.isin(ZERO_OVERHEAD_ENGS), "Overhead"] = 0.0

        # Management: zero overhead and zero wage
        mgmt_shift_mask = eng_shift.isin(MANAGEMENT_ENGINEERS)
        shift_totals.loc[mgmt_shift_mask, "Overhead"]          = 0.0
        shift_totals.loc[mgmt_shift_mask, "Day Basic Wage"]    = 0.0
        shift_totals.loc[mgmt_shift_mask, "Day Overtime Wage"] = 0.0
        shift_totals.loc[mgmt_shift_mask, "Total Pay"]         = 0.0
        shift_totals.loc[mgmt_shift_mask, "Wage/Pension/NI"]  = 0.0

        # --- Hourly rate (with rate change history) ---
        is_weekend  = shift_totals["Shift Start"].dt.weekday >= 5
        hourly_rate = shift_totals["Engineer"].map(ENGINEER_RATE_WEEKDAY).copy()
        weekend_rate = shift_totals["Engineer"].map(ENGINEER_RATE_WEEKEND)
        hourly_rate[is_weekend & weekend_rate.notna()] = weekend_rate[is_weekend & weekend_rate.notna()]
        hourly_rate = hourly_rate.fillna(0)

        for eng, changes in RATE_CHANGES.items():
            for eff_date, new_weekday, new_weekend in sorted(changes, key=lambda x: x[0]):
                m = (eng_shift == eng) & (shift_date >= eff_date)
                hourly_rate.loc[m & (~is_weekend)] = float(new_weekday)
                hourly_rate.loc[m & is_weekend]    = float(new_weekend)

        total_duration = (
            (shift_totals["Shift End"] - shift_totals["Shift Start"])
            .dt.total_seconds() / 3600
        ).fillna(0).clip(lower=0)

        shift_totals["Shift Hours"] = total_duration.round(2)

        # --- Initialise wage columns ---
        shift_totals["Day Basic Wage"]   = 0.0
        shift_totals["Day Overtime Wage"] = 0.0
        shift_totals["Total Pay"]         = 0.0
        shift_totals["Wage/Pension/NI"]   = 0.0

        # --- Subcontractor pay: £90 first hour, £60/hr after in 15-min increments ---
        sc_mask     = eng_shift.isin(SUB_CONTRACTORS)
        non_sc_mask = ~sc_mask

        if sc_mask.any():
            sc_hours            = shift_totals.loc[sc_mask, "Day Hours"].fillna(0)
            first_hour_charge   = 90 * (sc_hours > 0).astype(int)
            extra_hours_rounded = (np.ceil((sc_hours - 1).clip(lower=0) / 0.25) * 0.25).round(2)
            sc_total_pay        = first_hour_charge + extra_hours_rounded * 60

            shift_totals.loc[sc_mask, "Day Basic Wage"]    = sc_total_pay
            shift_totals.loc[sc_mask, "Total Pay"]         = sc_total_pay
            shift_totals.loc[sc_mask, "Wage/Pension/NI"]   = sc_total_pay

        # --- Standard employee pay ---
        home_travel_duration = (
            (shift_totals["Shift End"] - shift_totals["Last Time off Site"])
            .dt.total_seconds() / 3600
        ).fillna(0).clip(lower=0)

        pre_home_duration = (
            (shift_totals["Last Time off Site"] - shift_totals["Shift Start"])
            .dt.total_seconds() / 3600
        ).fillna(0).clip(lower=0)

        on_site_mask  = non_sc_mask & eng_shift.isin(ON_SITE_PAY_ENGINEERS)
        weekday_mask  = non_sc_mask & (~is_weekend) & (~on_site_mask)
        weekend_mask  = non_sc_mask & is_weekend    & (~on_site_mask)

        basic_hours    = pd.Series(0.0, index=shift_totals.index)
        overtime_hours = pd.Series(0.0, index=shift_totals.index)
        extra_drive    = pd.Series(0.0, index=shift_totals.index)

        # Weekdays: 9-hour basic (10 for TEN_HOUR_ENGINEERS), overtime after that
        weekday_active = weekday_mask & (total_duration > 0)
        basic_hours.loc[weekday_active] = 9.0
        basic_hours.loc[weekday_active & eng_shift.isin(TEN_HOUR_ENGINEERS)] = 10.0

        weekday_overtime = (pre_home_duration - 9).clip(lower=0)
        weekday_overtime.loc[total_duration <= 9] = 0.0
        overtime_hours.loc[weekday_mask] = weekday_overtime.loc[weekday_mask]

        weekday_extra_drive = (total_duration - 9).clip(lower=0).clip(upper=home_travel_duration)
        weekday_extra_drive.loc[total_duration <= 9] = 0.0
        extra_drive.loc[weekday_mask] = weekday_extra_drive.loc[weekday_mask]

        # Weekends: all pay in overtime at flat rate (1.0x), 5-hour minimum retainer
        basic_hours.loc[weekend_mask]    = 0.0
        overtime_hours.loc[weekend_mask] = shift_totals.loc[weekend_mask, "Day Hours"].fillna(0).clip(lower=WEEKEND_RETAINER_HOURS)

        # Flat rate on-site engineers (e.g. Jamie Boyd): on-site hours only, 7-hour retainer, no overtime multiplier
        basic_hours.loc[on_site_mask]    = shift_totals.loc[on_site_mask, "Day Hours"].fillna(0).clip(lower=ON_SITE_RETAINER_HOURS)
        overtime_hours.loc[on_site_mask] = 0.0

        overtime_factor = pd.Series(1.5, index=shift_totals.index)
        overtime_factor.loc[weekend_mask]  = 1.0
        overtime_factor.loc[on_site_mask]  = 1.0

        shift_totals.loc[non_sc_mask, "Day Basic Wage"] = (
            basic_hours[non_sc_mask] * hourly_rate[non_sc_mask]
        ).round(2)

        shift_totals.loc[non_sc_mask, "Day Overtime Wage"] = (
            overtime_hours[non_sc_mask] * hourly_rate[non_sc_mask] * overtime_factor[non_sc_mask]
            + extra_drive[non_sc_mask]  * hourly_rate[non_sc_mask]
        ).round(2)

        shift_totals.loc[non_sc_mask, "Total Pay"] = (
            shift_totals.loc[non_sc_mask, "Day Basic Wage"]
            + shift_totals.loc[non_sc_mask, "Day Overtime Wage"]
        ).round(2)

        shift_totals.loc[non_sc_mask, "Wage/Pension/NI"] = (
            (shift_totals.loc[non_sc_mask, "Day Basic Wage"] * 0.03
             + shift_totals.loc[non_sc_mask, "Total Pay"]) * 1.1435
        ).round(2)

        # --- Join shift totals back onto row-level data ---
        df = df.join(shift_totals[[
            "Day Cost", "Day Sell", "Day Labour", "Day Hours", "Day Part Profit",
            "Day Basic Wage", "Day Overtime Wage", "Total Pay", "Wage/Pension/NI",
            "Overhead", "Shift Hours", "First Job to Last Job Hours",
            "Shift First Job Travel", "Shift First Time on Site",
            "Shift Last Time off Site", "Shift Home Time", "Pay Month",
        ]], on="Shift ID")

        df["Overhead without Wage"] = pd.NA
        df["Total Cost"]            = pd.NA

        # Summary row = last row of each shift (carries the day-level totals)
        summary_idx  = df.groupby("Shift ID").tail(1).index
        mask_summary = df.index.isin(summary_idx)

        # Overhead on summary line
        df.loc[mask_summary, "Overhead without Wage"] = OVERHEAD_VALUE
        special_mask = mask_summary & df["Engineer"].astype(str).str.strip().isin(SPECIAL_ENGS)
        df.loc[special_mask, "Overhead without Wage"] = SPECIAL_OVERHEAD_VALUE

        eng_row  = df["Engineer"].astype(str).str.strip()
        row_date = _get_row_date(df)

        asst_row = eng_row.isin(ASSISTANTS)
        for name, cutoff in ASSISTANT_CUTOFFS.items():
            asst_row = asst_row & ~(eng_row.eq(name) & row_date.notna() & (row_date >= cutoff))

        df.loc[mask_summary & asst_row,                               "Overhead without Wage"] = 0.0
        df.loc[mask_summary & eng_row.isin(ZERO_OVERHEAD_ENGS),       "Overhead without Wage"] = 0.0
        mgmt_row_mask = mask_summary & eng_row.isin(MANAGEMENT_ENGINEERS)
        df.loc[mgmt_row_mask, "Overhead without Wage"]  = 0.0
        df.loc[mgmt_row_mask, "Day Basic Wage"]         = 0.0
        df.loc[mgmt_row_mask, "Day Overtime Wage"]      = 0.0
        df.loc[mgmt_row_mask, "Total Pay"]              = 0.0
        df.loc[mgmt_row_mask, "Wage/Pension/NI"]        = 0.0
        df.loc[special_mask, ["Day Basic Wage", "Day Overtime Wage", "Total Pay", "Wage/Pension/NI"]] = 0.0

        # --- Profit & margin ---
        df.loc[mask_summary, "Total Cost"] = (
            df.loc[mask_summary, "Wage/Pension/NI"].fillna(0)
            + df.loc[mask_summary, "Overhead without Wage"].fillna(0)
        ).round(2)

        df.loc[mask_summary, "Labour Profit"] = (
            df.loc[mask_summary, "Day Labour"].fillna(0)
            - df.loc[mask_summary, "Total Cost"].fillna(0)
        ).round(2)

        df.loc[mask_summary, "Labour Margin"] = (
            (df.loc[mask_summary, "Labour Profit"]
             / df.loc[mask_summary, "Day Labour"])
            .replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        ).round(2)

        df.loc[mask_summary, "Total Profit"] = (
            df.loc[mask_summary, "Labour Profit"].fillna(0)
            + df.loc[mask_summary, "Day Part Profit"].fillna(0)
        ).round(2)

        labour_profit   = df.loc[mask_summary, "Labour Profit"].fillna(0)
        parts_profit    = df.loc[mask_summary, "Day Part Profit"].fillna(0)
        labour_turnover = df.loc[mask_summary, "Day Labour"].fillna(0)
        part_sell       = df.loc[mask_summary, "Day Sell"].fillna(0)

        combined_margin = (
            (labour_profit + parts_profit) / (labour_turnover + part_sell)
        ).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        df.loc[mask_summary, "Combined Margin"] = combined_margin.round(2)

        # --- Bonus ---
        conditions = [
            df.loc[mask_summary, "Combined Margin"] <= 0,
            (df.loc[mask_summary, "Combined Margin"] > 0)  & (df.loc[mask_summary, "Combined Margin"] < 25),
            (df.loc[mask_summary, "Combined Margin"] >= 25) & (df.loc[mask_summary, "Combined Margin"] < 35),
            (df.loc[mask_summary, "Combined Margin"] >= 35) & (df.loc[mask_summary, "Combined Margin"] < 50),
             df.loc[mask_summary, "Combined Margin"] >= 50,
        ]
        df.loc[mask_summary, "Bonus"] = np.select(conditions, [-20, 0, 20, 50, 100], default=0)

        # No bonus for flat rate on-site engineers
        df.loc[mask_summary & eng_row.isin(ON_SITE_PAY_ENGINEERS), "Bonus"] = 0

        # No bonus for management
        df.loc[mask_summary & eng_row.isin(MANAGEMENT_ENGINEERS), "Bonus"] = 0

        # --- Office-visit-only shifts: zero overhead ---
        if {"Shift ID", "Job Type"}.issubset(df.columns):
            office_only = df["Job Type"].astype(str).str.strip().str.upper().eq("OFFICE VISIT")
            office_only_shift = office_only.groupby(df["Shift ID"]).transform("all")
            office_summary    = office_only_shift & mask_summary

            df.loc[office_only_shift, "Overhead"]                = 0
            df.loc[office_summary,    "Overhead without Wage"]   = 0
            df.loc[office_summary,    "Total Cost"]              = df.loc[office_summary, "Wage/Pension/NI"].fillna(0).round(2)
            df.loc[office_summary,    "Labour Profit"]           = (
                df.loc[office_summary, "Day Labour"].fillna(0)
                - df.loc[office_summary, "Total Cost"].fillna(0)
            ).round(2)

            office_combined = (
                (df.loc[office_summary, "Labour Profit"].fillna(0) + df.loc[office_summary, "Day Part Profit"].fillna(0))
                / (df.loc[office_summary, "Day Labour"].fillna(0) + df.loc[office_summary, "Day Sell"].fillna(0))
            ).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

            df.loc[office_summary, "Total Profit"] = (
                df.loc[office_summary, "Labour Profit"].fillna(0)
                + df.loc[office_summary, "Day Part Profit"].fillna(0)
            ).round(2)
            df.loc[office_summary, "Bonus"] = 0

        # --- Row Cost: allocate day's total cost across jobs by on-site hours ---
        if {"Shift ID", "_job_hours", "Total Cost"}.issubset(df.columns):
            shift_total_cost = df.groupby("Shift ID")["Total Cost"].transform("max").fillna(0)
            shift_hours      = df.groupby("Shift ID")["_job_hours"].transform("sum")
            df["Row Cost"]   = 0.0
            valid = shift_hours > 0
            df.loc[valid, "Row Cost"] = (
                shift_total_cost[valid] * df.loc[valid, "_job_hours"] / shift_hours[valid]
            ).round(2)
        else:
            df["Row Cost"] = pd.NA

        # --- Per-job profit ---
        if {"Labour", "Material Cost", "Material Sell", "Row Cost"}.issubset(df.columns):
            df["Labour Profit (Per Job)"] = (df["Labour"].fillna(0) - df["Row Cost"].fillna(0)).round(2)
            df["Parts Profit (Per Job)"]  = (df["Material Sell"].fillna(0) - df["Material Cost"].fillna(0)).round(2)
            df["Profit (Per Job)"]        = (df["Labour Profit (Per Job)"] + df["Parts Profit (Per Job)"]).round(2)
        else:
            df["Labour Profit (Per Job)"] = pd.NA
            df["Parts Profit (Per Job)"]  = pd.NA
            df["Profit (Per Job)"]        = pd.NA

        # --- Blank out summary-only columns on non-summary rows ---
        summary_only_cols = [
            "Day Cost", "Day Sell", "Day Labour", "Day Hours", "Real Date",
            "Day Part Profit", "Day Basic Wage", "Day Overtime Wage",
            "Overhead without Wage", "Total Cost", "Total Pay", "Wage/Pension/NI",
            "Shift Hours", "First Job to Last Job Hours",
            "Shift First Job Travel", "Shift First Time on Site",
            "Shift Last Time off Site", "Shift Home Time",
        ]
        for col in summary_only_cols:
            df.loc[~mask_summary, col] = pd.NA

        df = df.drop(columns=["Shift ID", "_job_hours"])

    # --- Ensure all expected columns exist ---
    for col in ["Overhead", "Day Cost", "Day Sell", "Day Labour", "Day Hours", "Real Date",
                "Day Part Profit", "Day Basic Wage", "Day Overtime Wage", "Total Pay",
                "Wage/Pension/NI", "Overhead without Wage", "Total Cost"]:
        if col not in df.columns:
            df[col] = pd.NA

    # --- Rename columns to final display names ---
    df = df.rename(columns={
        "Day Cost":             "Part Cost",
        "Day Sell":             "Part Sell",
        "Day Labour":           "Labour Turnover",
        "Overhead":             "Overhead Per Job",
        "Overhead without Wage":"Overhead",
        "Day Part Profit":      "Parts Profit",
        "Day Hours":            "On Site Hours",
    })

    # ==========================================================================
    # ROLE ASSIGNMENT
    # ==========================================================================
    ASSISTANTS_CLEAN    = {n.strip() for n in ASSISTANTS_FOR_ROLE}
    SUBCONTRACTORS_CLEAN = {n.strip() for n in SUBCONTRACTORS_FOR_ROLE}
    ENGINEERS_ALL       = {n.strip() for n in ENGINEER_RATE_WEEKDAY.keys()}
    ENGINEERS_CLEAN     = ENGINEERS_ALL - ASSISTANTS_CLEAN - SUBCONTRACTORS_CLEAN

    eng_clean = df["Engineer"].astype(str).str.strip()
    df["Role"] = "Unknown"
    df.loc[eng_clean.isin(ENGINEERS_CLEAN),       "Role"] = "Engineer"
    df.loc[eng_clean.isin(ASSISTANTS_CLEAN),      "Role"] = "Assistant"
    df.loc[eng_clean.isin(SUBCONTRACTORS_CLEAN),  "Role"] = "Sub Contractors"
    df.loc[eng_clean.isin(MANAGEMENT_ENGINEERS),  "Role"] = "Office"

    # Apply promotion dates: anyone in ASSISTANT_CUTOFFS transitions from Assistant to Engineer
    row_date = _get_row_date(df)
    for name, cutoff in ASSISTANT_CUTOFFS.items():
        m = eng_clean.eq(name) & row_date.notna()
        df.loc[m & (row_date >= cutoff), "Role"] = "Engineer"
        df.loc[m & (row_date < cutoff),  "Role"] = "Assistant"

    # ==========================================================================
    # ENGINEER RECALL LOGIC
    # ==========================================================================
    if {"Job Number", "Job Type", "Engineer"}.issubset(df.columns):
        job_type_up = df["Job Type"].astype(str).str.strip().str.upper()
        job_num_str = df["Job Number"].astype(str).str.strip()
        eng_clean   = df["Engineer"].astype(str).str.strip()
        row_date    = _get_row_date(df)

        is_asst = eng_clean.isin(ASSISTANTS)
        for name, cutoff in ASSISTANT_CUTOFFS.items():
            is_asst = is_asst & ~(eng_clean.eq(name) & row_date.notna() & (row_date >= cutoff))

        base_id    = job_num_str.str.split("/", n=1).str[0]
        rec_suffix = job_num_str.str.split("/", n=1).str[1]

        def parse_rec(x):
            if pd.isna(x): return -1
            x = str(x).strip()
            if x == "": return -1
            try: return int(x)
            except ValueError: return 9999

        rec_num = rec_suffix.apply(parse_rec)
        df["Engineer Recall"] = pd.NA

        tmp = df.assign(job_type_up=job_type_up, base_id=base_id,
                        rec_num=rec_num, is_assistant=is_asst)

        for base, idx in tmp.groupby("base_id").groups.items():
            grp          = tmp.loc[idx].sort_values("rec_num")
            last_engineer = None
            for row_idx, row in grp.iterrows():
                is_recall   = row["job_type_up"] == "RECALL"
                row_is_asst = bool(row["is_assistant"])
                eng         = row["Engineer"]
                if is_recall and last_engineer and not row_is_asst:
                    df.at[row_idx, "Engineer Recall"] = last_engineer
                if not row_is_asst and pd.notna(eng) and str(eng).strip():
                    last_engineer = eng
    else:
        df["Engineer Recall"] = pd.NA

    # ==========================================================================
    # SHIFT TYPE
    # ==========================================================================
    eng_clean = df["Engineer"].astype(str).str.strip()
    df["Shift Type"] = "Day"
    df.loc[eng_clean.isin(NIGHT_WORKERS), "Shift Type"] = "Night"

    df = df.drop(columns=["Total Cost per Job"], errors="ignore")

    # ==========================================================================
    # COLUMN ORDER
    # ==========================================================================
    desired_order = [
        "Job Number", "Quote Number", "Customer Order Number", "Job Type", "Status",
        "Job Category", "Customer Name", "Site Name", "Engineer",
        "Job Travel", "Time on Site", "Time off Site", "Home Time", "Real Date",
        "Material Cost", "Material Sell", "Labour", "Total Sell",
        "On Site Hours", "Shift Hours", "First Job to Last Job Hours",
        "Overhead Per Job", "Day Basic Wage", "Day Overtime Wage", "Total Pay",
        "Wage/Pension/NI", "Overhead", "Total Cost",
        "Part Cost", "Part Sell", "Parts Profit",
        "Labour Turnover", "Labour Profit", "Labour Margin",
        "Combined Margin", "Total Profit", "Bonus",
        "Row Cost", "Labour Profit (Per Job)", "Part Profit (Per Job)", "Profit (Per Job)",
    ]
    df = df[[c for c in desired_order if c in df.columns]
            + [c for c in df.columns if c not in desired_order]]

    return df


# ==============================================================================
# FTP HELPERS
# ==============================================================================
def connect_ftp() -> FTP_TLS:
    if not FTP_USER or not FTP_PASS:
        raise RuntimeError("FTP_USER and FTP_PASS must be set")
    ftps = FTP_TLS(FTP_HOST)
    ftps.login(FTP_USER, FTP_PASS)
    ftps.prot_p()
    return ftps


def ensure_dir(ftps: FTP_TLS, path: str) -> None:
    """Ensure a directory path exists on the FTP server, creating it if needed."""
    original = ftps.pwd()
    for part in [p for p in path.strip("/").split("/") if p]:
        try:
            ftps.cwd(part)
        except error_perm:
            ftps.mkd(part)
            ftps.cwd(part)
    ftps.cwd(original)


def list_csv_files(ftps: FTP_TLS, path: str) -> list[str]:
    """Return a list of .csv filenames in the given FTP directory."""
    ftps.cwd(path)
    try:
        names = ftps.nlst()
    except error_perm as e:
        if "No files found" in str(e):
            return []
        raise
    return [n for n in names if n.lower().endswith(".csv")]


def download_csv_to_dataframe(ftps: FTP_TLS, directory: str, filename: str) -> pd.DataFrame:
    """Download a CSV from FTP and return it as a DataFrame."""
    ftps.cwd(directory)
    buf = io.BytesIO()
    ftps.retrbinary(f"RETR {filename}", buf.write)
    buf.seek(0)
    return pd.read_csv(buf)


def upload_dataframe_as_csv(ftps: FTP_TLS, directory: str, filename: str, df: pd.DataFrame) -> None:
    """Upload a DataFrame as a CSV to FTP."""
    ftps.cwd(directory)
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    ftps.storbinary(f"STOR {filename}", buf)


def delete_file(ftps: FTP_TLS, directory: str, filename: str) -> None:
    """Delete a file from FTP, logging a warning if it fails."""
    try:
        ftps.cwd(directory)
        ftps.delete(filename)
        print(f"Deleted original file: {filename}")
    except Exception as e:
        print(f"Warning: Could not delete {filename}: {e}")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
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
            base, _ = os.path.splitext(name)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_name = f"{base}_clean_{ts}.csv"

            print(f"Processing {name} -> {processed_name}")
            df_raw   = download_csv_to_dataframe(ftps, INPUT_DIR, name)
            df_clean = (transform_parts_dataframe(df_raw)
                        if "required" in name.lower()
                        else transform_dataframe(df_raw))

            upload_dataframe_as_csv(ftps, OUTPUT_DIR, processed_name, df_clean)
            print(f"Uploaded cleaned file to {OUTPUT_DIR}/{processed_name}")

            local_path = os.path.join("/tmp", processed_name)
            df_clean.to_csv(local_path, index=False)
            print(f"Saved local file for drive upload: {local_path}")

            upload_to_drive(local_path, drive_filename=processed_name)
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
