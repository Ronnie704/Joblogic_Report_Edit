import io
import os
from ftplib import FTP_TLS, error_perm
import pandas as pd
import numpy as np

from upload_to_drive import upload_to_drive

FTP_HOST = "Ronnie789.synology.me"
FTP_USER = os.environ.get("FTP_USER")
FTP_PASS = os.environ.get("FTP_PASS")

INPUT_DIR = "/JoblogicFTP"
OUTPUT_DIR = "/JoblogicFTP/processed"

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
    "Jamie Scott": 13,
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
    "Iosua Caloro": 12.50,
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
    "Gavain Brown ": 35,
    "Greg Czubak": 0,
    "Jair Gomes": 35,
    "Jake LeBeau": 25,
    "Jamie Scott": 25,
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
    "William Mcmillan ": 35,
    "Younas": 35,
    "kieran Mbala": 35,
}

#
def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up the Joblogic export:
    - drop unwanted columns
    - PPM: if Total Sell == 0, use Job Ref 1 as Total Sell
    - Quoted: Total Sell = Total Sell - Material Sell
    - Labour = Total Sell - Material Sell
    - sort by Engineer
    """
    if "Status" in df.columns:
        df = df.loc[df["Status"].astype(str).str.strip().str.upper() != "CANCELLED"].copy()

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

    if {"Job Travel", "Time on Site"}.issubset(df.columns):
        mask = df["Job Travel"].isna() & df["Time on Site"].notna()
        df.loc[mask, "Job Travel"] = df.loc[mask, "Time on Site"]

    # Sort by Engineer (A–Z)
    if {"Engineer", "Job Travel"}.issubset(df.columns):
        df = df.sort_values(
            by=["Engineer", "Job Travel"],
            ascending=[True, True]
        ).reset_index(drop=True)
    elif "Engineer" in df.columns:
        df = df.sort_values(by="Engineer", ascending=True).reset_index(drop=True)

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

        end_of_day = gap.dt.total_seconds() > 8 * 3600
        mask = mask & end_of_day

        df.loc[mask, "Home Time"] = df.loc[mask, "Time off Site"]

        df = df.drop(columns=["next_travel"])

    # 3. If Job Type == 'PPM' and Total Sell == 0, use Job Ref 1 as new Total Sell
    if {"Job Type", "Total Sell", "Job Ref 1"}.issubset(df.columns):
        condition_ppm = (
            df["Job Type"].str.strip().str.upper() == "PPM"
        ) & (df["Total Sell"].fillna(0) == 0)
        df.loc[condition_ppm, "Total Sell"] = df.loc[condition_ppm, "Job Ref 1"]

    # 4. If Job Type == 'Quoted' and Job Number starts with 'Q0',
    #    set Total Sell = Total Sell - Material Sell.
    #    If it starts with 'QR', leave Total Sell as-is.
    if {"Job Type", "Total Sell", "Material Sell", "Job Number"}.issubset(df.columns):
        job_type = df["Job Type"].astype(str).str.strip().str.upper()
        job_num  = df["Job Number"].astype(str).str.strip().str.upper()

        is_quoted = job_type.eq("QUOTED")        # this matches "Quoted   " too
        is_q0     = job_num.str.startswith("Q0") # only fix these

        mask_fix = is_quoted & is_q0

        df.loc[mask_fix, "Total Sell"] = (
            df.loc[mask_fix, "Total Sell"].fillna(0)
            - df.loc[mask_fix, "Material Sell"].fillna(0)
        )

    # 5. Drop Job Ref 1 now (we’ve used it)
    if "Job Ref 1" in df.columns:
        df = df.drop(columns=["Job Ref 1"], errors="ignore")

    # 6. Add Labour column (always = Total Sell - Material Sell)
    if {"Total Sell", "Material Sell"}.issubset(df.columns):
        df["Labour"] = df["Total Sell"].fillna(0) - df["Material Sell"].fillna(0)
    else:
        df["Labour"] = pd.NA

   

    #8 calculate day cost
    if {"Engineer", "Job Travel", "Home Time", "Material Cost", "Material Sell"}.issubset(df.columns):
        # sort once
        df = df.sort_values(by=["Engineer", "Job Travel"]).reset_index(drop=True)

        # --- REAL DATE & SHIFT ID (handles night shifts correctly, using Time off Site) ---

        jt = df["Job Travel"]

        if "Time off Site" in df.columns:
            # previous Time off Site per engineer
            prev_off = df.groupby("Engineer")["Time off Site"].shift(1)
            # gap in hours since previous Time off Site
            gap_hours = (jt - prev_off).dt.total_seconds() / 3600
        else:
            # if no Time off Site, we can't measure the gap
            gap_hours = pd.Series(np.nan, index=df.index)

        # treat 00:00–05:59 as "early"
        early = jt.dt.hour < 6

        # long rest = first job OR >= 8 hours since last Time off Site
        long_rest = gap_hours.isna() | (gap_hours >= 8)

        # only continue previous day if early AND not a long rest
        use_prev_day = early & (~long_rest)

        # start from calendar date of Job Travel
        real_date = jt.dt.date

        # for true night continuation rows, move to previous calendar day
        real_date[use_prev_day] = (jt[use_prev_day] - pd.Timedelta(days=1)).dt.date

        df["Real Date"] = real_date
        df["Real Date (Each Row)"] = df["Real Date"]

        # one shift per engineer per Real Date
        df["Shift ID"] = (
            df["Engineer"].astype(str).str.strip()
            + "_"
            + df["Real Date"].astype(str)
        )

        if {"Time on Site", "Time off Site"}.issubset(df.columns):
            duration = df["Time off Site"] - df["Time on Site"]
            df["_job_hours"] = (duration.dt.total_seconds() / 3600).fillna(0)
        else:
            df["_job_hours"] = 0.0
            
        #----------------------------------------------------------------
        if {
            "Job Number", "Engineer", "Time on Site", "Time off Site",
            "Material Cost", "Material Sell", "Labour", "Total Sell"
        }.issubset(df.columns):

            # 1️⃣ Hours per visit (row)
            df["_job_hours_split"] = (
                (df["Time off Site"] - df["Time on Site"])
                .dt.total_seconds() / 3600
            ).fillna(0)

            # 2️⃣ Total hours per job (all engineers, all visits)
            job_total_hours = df.groupby("Job Number")["_job_hours_split"].transform("sum")

            # 3️⃣ Total hours per engineer per job (sum of all their visits)
            eng_job_hours = df.groupby(["Job Number", "Engineer"])["_job_hours_split"].transform("sum")

            # 4️⃣ Engineer share of the job (same as before)
            engineer_share = np.where(
                job_total_hours > 0,
                eng_job_hours / job_total_hours,
                0.0,
            )

            # 5️⃣ Within-engineer share per row (visit/day)
            row_within_engineer_share = np.where(
                eng_job_hours > 0,
                df["_job_hours_split"] / eng_job_hours,
                0.0,
            )

        # 6️⃣ Final row share = engineer share × within-engineer share
        #    (so: first between engineers, then between their visits)
        row_share = engineer_share * row_within_engineer_share

        # make it a pandas Series so we can groupby on it
        row_share = pd.Series(row_share, index=df.index, dtype=float)

        # ---------- ASSISTANT LOGIC ----------
        ASSISTANTS = {
            "Airon Paul",
            "Arron Barnes",
            "Iosua Caloro",
            "Jair Gomes",
            "Jake LeBeau",
            "Jamie Scott",
            "Jordan Utter",
        }

        eng_clean = df["Engineer"].astype(str).str.strip()
        is_assistant = eng_clean.isin(ASSISTANTS)

        # per-job: does this job have at least one non-assistant?
        has_main = (~is_assistant).groupby(df["Job Number"]).transform("any")

        # start from original row_share
        row_share_adj = row_share.copy()

        # if job has a main engineer, assistants get 0 share
        mask_assist_zero = is_assistant & has_main
        row_share_adj[mask_assist_zero] = 0.0

        # renormalise shares per job so totals still match
        sum_shares = row_share_adj.groupby(df["Job Number"]).transform("sum")

        row_share_final = row_share_adj.copy()
        renorm_mask = has_main & (sum_shares > 0)

        row_share_final[renorm_mask] = (
            row_share_adj[renorm_mask] / sum_shares[renorm_mask]
        )
        # ---------- END ASSISTANT LOGIC ----------

        # 7️⃣ Apply final row share to value columns
        for col in ["Material Cost", "Material Sell", "Labour", "Total Sell"]:
            if col in df.columns:
                df[col] = (df[col].fillna(0) * row_share_final).round(2)

        # 8️⃣ Clean up helper column
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
                "Time on Site": "min",
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
                "Time on Site": "First Time on Site",
            })
        )

        shift_totals["Shift First Job Travel"] = df.groupby("Shift ID")["Job Travel"].transform("min").groupby(df["Shift ID"]).first()
        shift_totals["Shift First Time on Site"] = df.groupby("Shift ID")["Time on Site"].transform("min").groupby(df["Shift ID"]).first()
        shift_totals["Shift Last Time off Site"] = df.groupby("Shift ID")["Time off Site"].transform("max").groupby(df["Shift ID"]).first()
        shift_totals["Shift Home Time"] = df.groupby("Shift ID")["Home Time"].transform("max").groupby(df["Shift ID"]).first()

        shift_totals["First Job to Last Job Hours"] = (
            (shift_totals["Last Time off Site"] - shift_totals["First Time on Site"])
            .dt.total_seconds() / 3600
        ).fillna(0).round(2)
        
        shift_totals["Day Part Profit"] = shift_totals["Day Sell"] - shift_totals["Day Cost"]

        from datetime import date, timedelta

        def compute_pay_month(day):
            if pd.isna(day):
                return pd.NA

            d = pd.to_datetime(day).date()
            y, m = d.year, d.month

            # --- end of pay month: last Monday in month with Tue+Wed still in month ---
            if m == 12:
                first_next = date(y + 1, 1, 1)
            else:
                first_next = date(y, m + 1, 1)
            last_day = first_next - timedelta(days=1)

            # start from last_day - 2 so Mon has Tue+Wed still in the same month
            cur = last_day - timedelta(days=2)
            end = None
            while cur.month == m:
                if cur.weekday() == 0:  # Monday
                    end = cur
                    break
                cur -= timedelta(days=1)
            if end is None:
                # fallback: just last Monday in month
                cur = last_day
                while cur.month == m and cur.weekday() != 0:
                    cur -= timedelta(days=1)
                end = cur

            # --- previous month y, m-1 ---
            if m == 1:
                py, pm = y - 1, 12
            else:
                py, pm = y, m - 1

            if pm == 12:
                prev_first_next = date(py + 1, 1, 1)
            else:
                prev_first_next = date(py, pm + 1, 1)
            prev_last = prev_first_next - timedelta(days=1)

            # start = last Tuesday in previous month with a Wednesday still in that month
            cur2 = prev_last - timedelta(days=1)
            start = None
            while cur2.month == pm:
                if cur2.weekday() == 1 and (cur2 + timedelta(days=1)).month == pm:  # Tuesday & next day still in month
                    start = cur2
                    break
                cur2 -= timedelta(days=1)
            if start is None:
                # fallback: last Tuesday of previous month
                cur2 = prev_last
                while cur2.month == pm and cur2.weekday() != 1:
                    cur2 -= timedelta(days=1)
                start = cur2

            # --- decide which pay month this date belongs to ---
            if d < start:
                # belongs to pay month BEFORE previous month
                if pm == 1:
                    yy, mm = py - 1, 12
                else:
                    yy, mm = py, pm - 1
            elif d > end:
                # belongs to next pay month
                if m == 12:
                    yy, mm = y + 1, 1
                else:
                    yy, mm = y, m + 1
            else:
                # belongs to current calendar month
                yy, mm = y, m

            return f"{yy}-{mm:02d}"

        # make sure Real Date exists for shifts (from Shift Start)
        shift_totals["Real Date"] = shift_totals["Shift Start"].dt.date
        shift_totals["Pay Month"] = shift_totals["Real Date"].apply(compute_pay_month)
            
        
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

        #------------ Zero Overhead Engineers -------------------
        ZERO_OVERHEAD_ENGS = {"Chris Eland"}
        zero_overhead_shift_mask = shift_totals["Engineer"].astype(str).str.strip().isin(ZERO_OVERHEAD_ENGS)
        shift_totals.loc[zero_overhead_shift_mask, "Overhead"] = 0.0
            
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

        shift_totals["Shift Hours"] = total_duration.round(2)

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

        df = df.join(shift_totals[["Day Cost", "Day Sell", "Day Labour", "Day Hours", "Day Part Profit", "Day Basic Wage", "Day Overtime Wage", "Total Pay", "Wage/Pension/NI", "Overhead", "Shift Hours", "First Job to Last Job Hours", "Shift First Job Travel", "Shift First Time on Site", "Shift Last Time off Site", "Shift Home Time", "Pay Month",]], on="Shift ID")

        df["Overhead without Wage"] = pd.NA 
        df["Total Cost"] = pd.NA
        summary_idx = df.groupby("Shift ID").tail(1).index
        mask_summary = df.index.isin(summary_idx)

        df.loc[mask_summary, "Overhead without Wage"] = OVERHEAD_VALUE
        special_mask = mask_summary & df["Engineer"].astype(str).str.strip().isin(SPECIAL_ENGS)
        df.loc[special_mask, "Overhead without Wage"] = 600.0

        zero_overhead_row_mask = mask_summary & df["Engineer"].astype(str).str.strip().isin(ZERO_OVERHEAD_ENGS)
        df.loc[zero_overhead_row_mask, "Overhead without Wage"] = 0.0

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
            (df.loc[mask_summary, "Combined Margin"] > 0) & (df.loc[mask_summary, "Combined Margin"] <25),
            (df.loc[mask_summary, "Combined Margin"] >= 25) & (df.loc[mask_summary, "Combined Margin"] < 35),
            (df.loc[mask_summary, "Combined Margin"] >= 35) & (df.loc[mask_summary, "Combined Margin"] < 50),
            df.loc[mask_summary, "Combined Margin"] >=50,
        ]
        choices = [-20, 0, 20, 50, 100]

        df.loc[mask_summary, "Bonus"] = np.select(conditions, choices, default=0)

        # If Only Office Visit in a day 
        if {"Shift ID", "Job Type"}.issubset(df.columns):
            status_upper = df["Job Type"].astype(str).str.strip().str.upper()
            office_rows = status_upper.eq("OFFICE VISIT")

            office_only_shift = office_rows.groupby(df["Shift ID"]).transform("all")

            df.loc[office_only_shift, "Overhead"] = 0

            office_summary = office_only_shift & mask_summary

            df.loc[office_summary, "Overhead without Wage"] = 0

            df.loc[office_summary, "Total Cost"] = (
                df.loc[office_summary, "Wage/Pension/NI"].fillna(0)
            ).round(2)

            df.loc[office_summary, "Labour Profit"] = (
                df.loc[office_summary, "Day Labour"].fillna(0)
                - df.loc[office_summary, "Total Cost"].fillna(0)
            ).round(2)

            labour_profit_off = df.loc[office_summary, "Labour Profit"].fillna(0)
            parts_profit_off = df.loc[office_summary, "Day Part Profit"].fillna(0)
            labour_turnover_off = df.loc[office_summary, "Day Labour"].fillna(0)
            part_sell_off = df.loc[office_summary, "Day Sell"].fillna(0)

            combined_margin_off = (
                (labour_profit_off + parts_profit_off)
                / (labour_turnover_off + part_sell_off)
            )
            combined_margin_off = combined_margin_off.replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0) *100

            df.loc[office_summary, "Total Profit"] = (
                labour_profit_off + parts_profit_off
            ).round(2)

            df.loc[office_summary, "Bonus"] = 0
        #------------------------------------------------------------------------------
        # -------------------- ROW COST (time-based, includes office rows) --------------------

        if {"Shift ID", "_job_hours", "Total Cost"}.issubset(df.columns):

            # same Total Cost for every row in a shift
            shift_total_cost = (
                df.groupby("Shift ID")["Total Cost"]
                  .transform("max")
                  .fillna(0)
            )

            # total hours per shift (sum of on-site hours)
            shift_hours = (
                df.groupby("Shift ID")["_job_hours"]
                  .transform("sum")
            )

            df["Row Cost"] = 0.0
            valid = shift_hours > 0
            df.loc[valid, "Row Cost"] = (
                shift_total_cost[valid] * df.loc[valid, "_job_hours"] / shift_hours[valid]
            ).round(2)
        else:
            df["Row Cost"] = pd.NA
        # -------------------------------------------------------------------------------------
        #Per Job Profit
        if {"Labour", "Material Cost", "Material Sell", "Row Cost"}.issubset(df.columns):
            df["Labour Profit (Per Job)"] = (
                df["Labour"].fillna(0) - df["Row Cost"].fillna(0)
            ).round(2)

            df["Parts Profit (Per Job)"] = (
                df["Material Sell"].fillna(0) - df["Material Cost"].fillna(0)
            ).round(2)

            df["Profit (Per Job)"] = (
                df["Labour Profit (Per Job)"]
                + df["Parts Profit (Per Job)"]
            ).round(2)
        else:
            df["Labour Profit (Per Job)"] = pd.NA
            df["Parts Profit (Per Job)"] = pd.NA
            df["Profit (Per Job)"] = pd.NA
        

        #------------------------------------------------------------------------------

        for col in ["Day Cost", "Day Sell", "Day Labour", "Day Hours", "Real Date","Day Part Profit", "Day Basic Wage", "Day Overtime Wage", "Overhead without Wage", "Total Cost", "Total Pay", "Wage/Pension/NI", "Shift Hours", "First Job to Last Job Hours", "Shift First Job Travel", "Shift First Time on Site", "Shift Last Time off Site", "Shift Home Time",]:
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
        df["Shift Hours"] = pd.NA
        

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
        "Day Part Profit": "Parts Profit",
        "Day Hours": "On Site Hours",
    })

    ASSISTANTS_FOR_ROLE = {
        "Airon Paul",
        "Arron Barnes",
        "Iosua Caloro",
        "Jair Gomes",
        "Jake LeBeau",
        "Jamie Scott",
        "Jordan Utter",
    }

    SUBCONTRACTORS_FOR_ROLE = {
        "Kevin Aubignac",
        "Ellis Russell",
        "Greg Czubak",
        "Mike Weare",
    }

    ASSISTANTS_CLEAN = {name.strip() for name in ASSISTANTS_FOR_ROLE}
    SUBCONTRACTORS_CLEAN = {name.strip() for name in SUBCONTRACTORS_FOR_ROLE}
    ENGINEERS_ALL = {name.strip() for name in ENGINEER_RATE_WEEKDAY.keys()}
    ENGINEERS_CLEAN = ENGINEERS_ALL -ASSISTANTS_CLEAN - SUBCONTRACTORS_CLEAN
    eng_clean = df["Engineer"].astype(str).str.strip()

    df["Role"] = "Unknown"

    df.loc[eng_clean.isin(ENGINEERS_CLEAN), "Role"] = "Engineer"
    df.loc[eng_clean.isin(ASSISTANTS_CLEAN), "Role"] = "Assistant"
    df.loc[eng_clean.isin(SUBCONTRACTORS_CLEAN), "Role"] = "Sub Contractors"

    #-------------Engineer Recall Logic----------------
    if {"Job Number", "Job Type", "Engineer"}.issubset(df.columns):

        # Normalise
        job_type_up = df["Job Type"].astype(str).str.strip().str.upper()
        job_num_str = df["Job Number"].astype(str).str.strip()
        eng_clean   = df["Engineer"].astype(str).str.strip()

        # Assistants mask (same list as in the cost-sharing logic)
        is_assistant = eng_clean.isin(ASSISTANTS)

        # Base id = bit before "/", e.g. "ABC/000" -> "ABC"
        base_id = job_num_str.str.split("/", n=1).str[0]
        # Suffix = bit after "/", e.g. "ABC/000" -> "000"
        rec_suffix = job_num_str.str.split("/", n=1).str[1]

        # Turn suffix into an integer so sort goes: original (-1), /000, /001, /002...
        def parse_rec(x):
            if pd.isna(x):
                return -1  # original job (no slash) always first
            x = str(x).strip()
            if x == "":
                return -1
            try:
                return int(x)
            except ValueError:
                # weird suffix, push it to the end
                return 9999

        rec_num = rec_suffix.apply(parse_rec)

        # Make sure the column exists
        df["Engineer Recall"] = pd.NA

        # Helper frame for grouping
        tmp = df.copy()
        tmp["job_type_up"]  = job_type_up
        tmp["base_id"]      = base_id
        tmp["rec_num"]      = rec_num
        tmp["is_assistant"] = is_assistant

        for base, idx in tmp.groupby("base_id").groups.items():
            grp = tmp.loc[idx].sort_values("rec_num")

            last_engineer = None  # last *non-assistant* engineer in this chain

            for row_idx, row in grp.iterrows():
                jt            = row["job_type_up"]
                eng           = row["Engineer"]
                row_is_asst   = bool(row["is_assistant"])

                # If this is a RECALL row and we have a previous non-assistant engineer
                # AND this row itself is not an assistant, then fill Engineer Recall.
                if jt == "RECALL" and (last_engineer is not None) and (not row_is_asst):
                    df.at[row_idx, "Engineer Recall"] = last_engineer

                # Update last_engineer ONLY from non-assistant rows
                if (not row_is_asst) and pd.notna(eng) and str(eng).strip() != "":
                    last_engineer = eng
    else:
        df["Engineer Recall"] = pd.NA
    #--------------------------------------------------

    NIGHT_WORKERS = {
        "Adrian Lewis",
        "Airon Paul",
        "Bernard Bezuidenhout",
        "Fabio Conceiocoa",
        "Gavain Brown",
        "Jair Gomes",
        "Jamie Scott",
        "Jordan Utter",
        "Mike Weare",
        "Nelson Vieira",
        "Sharick Bartley",
        "Younas",
    }

    eng_clean = df["Engineer"].astype(str).str.strip()

    df["Shift Type"] = "Day"
    df.loc[eng_clean.isin(NIGHT_WORKERS), "Shift Type"] = "Night"

    cols_to_remove = [
        "Total Cost per Job",
    ]
    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors="ignore")

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
        "On Site Hours",
        "Shift Hours",
        "First Job to Last Job Hours",
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
        "Row Cost",
        "Labour Profit (Per Job)",
        "Part Profit (Per Job)",
        "Profit (Per Job)",
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
  
