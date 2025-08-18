import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------
# Hudl Tendency Analyzer (Universal Language)
# Streamlit one-file app
# -----------------------
# Usage:
#   1) Save as app.py
#   2) pip install streamlit pandas numpy
#   3) streamlit run app.py
# -----------------------

st.set_page_config(page_title="Hudl Tendency Analyzer", layout="wide")
st.title("🏈 Hudl Tendency Analyzer & Game-Plan Builder — Universal Terminology")
st.caption("Upload a Hudl-style CSV to get overall & situational tendencies, then auto-generate a game-plan using universal football language.")

# -----------------------
# Config & helpers
# -----------------------
TEMPLATE_COLS = [
    "PLAY_NUM","ODK","DN","DIST","YARD_LN","HASH","OFF_FORM","OFF_STR","BACKFIELD",
    "OFF_PLAY","PLAY_TYPE","PLAY_DIR","RESULT","GN_LS","EFF","DEF_FRONT","DEF_STR",
    "BLITZ","COVERAGE","QTR","MOTION","MOTION_DIR","PASSER","RECEIVER","PLAY_TYPE_RPS",
    "RUSHER","TEAM"
]

DIST_BUCKETS = [
    (0, 3, "short (1-3)"),
    (4, 6, "medium (4-6)"),
    (7, 10, "long (7-10)"),
    (11, 999, "very long (11+)"),
]

# ---- Universal offensive concepts (no team-specific play names) ----
OFF_PRESSURE_ANSWERS = [
    "RB/WR/TE screens",
    "quick game (slant/flat/hitch)",
    "hot throws vs pressure",
    "max-protect shot (7-man)"
]
OFF_MAN_BEATERS = [
    "Mesh (crossers)",
    "Rub/stack releases",
    "Slant/Flat & Option routes",
    "Back-shoulder fade"
]
OFF_C3_BEATERS = [
    "3-level Flood (Sail)",
    "Curl/Flat",
    "Dagger (seam+dig)",
    "Seam shots vs MOFC"
]
OFF_C4_BEATERS = [
    "Posts & benders",
    "Scissors (Post+Corner)",
    "Flood variations",
    "Deep over (play-action)"
]
OFF_SCREEN_FAMILY = [
    "RB screen",
    "WR bubble/tunnel",
    "TE screen"
]
RUN_GAME = [
    "Inside Zone",
    "Outside Zone / Stretch",
    "Power",
    "Counter",
    "Iso / Lead",
    "Trap",
    "Pin-pull / Toss"
]

# Optional quick reference (personnel & formations) for the UI
UNIVERSAL_PERSONNEL = [
    "10 (1 RB, 0 TE)", "11 (1 RB, 1 TE)", "12 (1 RB, 2 TE)",
    "20 (2 RB, 0 TE)", "21 (2 RB, 1 TE)", "22 (2 RB, 2 TE)", "Empty"
]
UNIVERSAL_BASE_ALIGN = [
    "I-Formation", "Split Backs / Pro", "Singleback",
    "Shotgun", "Pistol", "Trips", "Bunch", "Spread", "Empty"
]

@st.cache_data(show_spinner=False)
def template_csv_bytes() -> bytes:
    df = pd.DataFrame(columns=TEMPLATE_COLS)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def to_bucket_dist(dist):
    try:
        d = int(dist)
    except Exception:
        return "unknown"
    for lo, hi, name in DIST_BUCKETS:
        if lo <= d <= hi:
            return name
    return "unknown"


def normalize_play_type(row: pd.Series) -> str:
    val = str(row.get("PLAY_TYPE_RPS") or "").strip().upper()
    if val in {"R", "RUN"}:
        return "Run"
    if val in {"P", "PASS"}:
        return "Pass"
    if val in {"S", "SCREEN"}:
        return "Screen"
    t = str(row.get("PLAY_TYPE") or "").lower()
    if "screen" in t:
        return "Screen"
    if "pass" in t:
        return "Pass"
    if "run" in t or t in {"ko", "po", "rush"}:
        return "Run"
    return "Unknown"


def left_mid_right(s) -> str:
    # Handle NaN, numerics, and strings safely
    if pd.isna(s):
        return "Unknown"
    try:
        si = int(float(s))
        return { -1: "Left", 0: "Middle", 1: "Right" }.get(si, "Unknown")
    except Exception:
        pass
    s = str(s).strip().lower()
    if not s:
        return "Unknown"
    if s in {"l","lt","left"} or s.startswith("l"):
        return "Left"
    if s in {"r","rt","right"} or s.startswith("r"):
        return "Right"
    if s in {"m","mid","middle","inside","in","center","ctr"} or s.startswith(("m","i")):
        return "Middle"
    return "Unknown"


def red_zone_flag(yard_ln: float) -> str:
    if pd.isna(yard_ln):
        return "no"
    return "yes" if 1 <= yard_ln <= 20 else "no"


def field_zone(yard_ln: float) -> str:
    # yard line: negative = own, positive = opponent; 0 ~ midfield
    if pd.isna(yard_ln):
        return "unknown"
    yl = yard_ln
    if yl <= -90:
        return "own 10 and in"
    if -89 <= yl <= -21:
        return "own territory (21-49)"
    if -20 <= yl <= -1:
        return "near midfield (own 1-20)"
    if 0 <= yl <= 20:
        return "midfield (50 +/- 20)"
    if 21 <= yl <= 35:
        return "plus territory (35-21)"
    if 36 <= yl <= 50:
        return "scoring fringe (20-34)"
    if 1 <= yl <= 20:
        return "red zone (20 and in)"
    return "unknown"


def tendency_table(df: pd.DataFrame, dims, outcome_col="PLAY_TYPE_NORM"):
    # Robust % calc using transform so index aligns with g
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=dims + [outcome_col, "plays", "%"])
    g = df.groupby(dims + [outcome_col], dropna=False).size().reset_index(name="plays")
    denom = g.groupby(dims)["plays"].transform("sum").replace(0, np.nan)
    g["%"] = ((100 * g["plays"] / denom).round(1)).fillna(0)
    return g.sort_values(dims + ["plays"], ascending=[True]*len(dims) + [False])


def compute_blitz_rate(df: pd.DataFrame, dims):
    g = df.copy()
    g["BLITZ_FLAG"] = g["BLITZ"].astype(str).str.lower().isin(["1","y","yes","t","true"])
    tbl = g.groupby(dims)["BLITZ_FLAG"].agg(["mean","count"]).reset_index()
    tbl.rename(columns={"mean":"blitz_rate","count":"plays"}, inplace=True)
    tbl["blitz_rate%"] = (100 * tbl["blitz_rate"]).round(1)
    return tbl.sort_values(["plays"], ascending=False)


def compute_coverage(df: pd.DataFrame, dims):
    g = df.copy()
    g["COVERAGE_N"] = g["COVERAGE"].fillna("Unknown").astype(str).str.upper()
    tbl = g.groupby(dims + ["COVERAGE_N"]).size().reset_index(name="plays")
    tbl["%"] = tbl.groupby(dims)["plays"].apply(lambda x: (100 * x / x.sum()).round(1))
    return tbl.sort_values(dims + ["plays"], ascending=[True]*len(dims) + [False])


def safe_rate(n, d):
    return (n / d) if d else 0.0


def build_suggestions(overall, by_down, by_dist, by_form, blitz_3rd, cov_3rd, hash_tbl, motion_tbl):
    suggestions = []

    total = overall["plays"].sum() if len(overall) else 0
    rp = overall[overall["PLAY_TYPE_NORM"].isin(["Run","Pass","Screen"])]
    run_p = safe_rate(rp[rp["PLAY_TYPE_NORM"]=="Run"]["plays"].sum(), total)
    pass_p = safe_rate(rp[rp["PLAY_TYPE_NORM"]=="Pass"]["plays"].sum(), total)
    screen_p = safe_rate(rp[rp["PLAY_TYPE_NORM"]=="Screen"]["plays"].sum(), total)

    if run_p >= 0.60:
        suggestions.append(f"Run-heavy profile overall ({run_p:.0%}). Defensively: add an extra fitter on early downs. Offensively: plan play-action off base runs.")
    elif pass_p >= 0.60:
        suggestions.append(f"Pass-heavy profile overall ({pass_p:.0%}). Defensively: mix simulated pressures and rotate late from two-high shells.")
    if screen_p >= 0.10:
        suggestions.append(f"Screens show up {screen_p:.0%}. DL: retrace and affect throw lanes. Offense: keep screen menu ready (RB/WR/TE).")

    for d in [1,2,3]:
        sub = by_down[by_down["DN"]==d]
        if len(sub):
            r = sub[sub["PLAY_TYPE_NORM"]=="Run"]["%"].sum() if "Run" in set(sub["PLAY_TYPE_NORM"]) else 0
            p = sub[sub["PLAY_TYPE_NORM"]=="Pass"]["%"].sum() if "Pass" in set(sub["PLAY_TYPE_NORM"]) else 0
            if d==1 and r>=60:
                suggestions.append("First down leans run. Fit downhill and stay alert for early down play-action shots.")
            if d==3 and p>=70:
                suggestions.append("Third down leans pass. Defense: simulated pressure, play the sticks. Offense: quick man/zone beaters.")

    td = by_dist[(by_dist["DN"]==3)]
    if len(td):
        long_pass = td[(td["DIST_BUCKET"]=="long (7-10)") & (td["PLAY_TYPE_NORM"]=="Pass")]
        if len(long_pass) and long_pass["%"].iloc[0] >= 70:
            suggestions.append("3rd & 7–10 high pass tendency. Offense: 3-level Flood/Sail or Dagger. Defense: mug A-gaps, spin to 3-robber.")
        vl_pass = td[(td["DIST_BUCKET"]=="very long (11+)") & (td["PLAY_TYPE_NORM"]=="Pass")]
        if len(vl_pass) and vl_pass["%"].iloc[0] >= 80:
            suggestions.append("3rd & 11+ = must-pass. Spy QB if mobile. Offense: shot plays off max protection vs soft zone.")

    if len(by_form):
        top_form = by_form.groupby(["OFF_FORM"])["plays"].sum().reset_index().sort_values("plays", ascending=False).head(1)
        if len(top_form):
            f = top_form["OFF_FORM"].iloc[0]
            f_tbl = by_form[by_form["OFF_FORM"]==f]
            run_bias = f_tbl[f_tbl["PLAY_TYPE_NORM"]=="Run"]["%"].sum() if "Run" in set(f_tbl["PLAY_TYPE_NORM"]) else 0
            if run_bias >= 65:
                suggestions.append(f"In {f}, strong run tendency (~{run_bias:.0f}%). Defensively: set edges, close interior gaps. Offensively: plan play-action counters from same look.")

    if len(blitz_3rd):
        high_blitz = blitz_3rd[blitz_3rd["blitz_rate"]>=0.35]
        if len(high_blitz):
            suggestions.append("They pressure on 3rd (≥35%). Offense: screens, quick game, and hot throws; consider max-protect shots.")

    if len(cov_3rd):
        top_cov = cov_3rd.groupby(["COVERAGE_N"])["plays"].sum().reset_index().sort_values("plays", ascending=False).head(1)
        if len(top_cov):
            cov = top_cov["COVERAGE_N"].iloc[0]
            if cov in {"COVER 1","C1","MAN","COVER1"}:
                suggestions.append("3rd-down = Man. Offense: Mesh, rub/stack releases, option routes, back-shoulder fades.")
            if cov in {"COVER 3","C3","THREE","COVER3"}:
                suggestions.append("3rd-down = Cover 3 (MOFC). Offense: Flood/Sail, Curl/Flat, Dagger; attack seams and curl window.")
            if cov in {"COVER 4","C4","QUARTERS"}:
                suggestions.append("3rd-down = Quarters. Offense: posts & benders, scissors (post+corner), deep overs.")

    if len(hash_tbl):
        left = hash_tbl[hash_tbl["HASH_N"]=="L"]["%"].sum() if any(hash_tbl["HASH_N"]=="L") else 0
        right = hash_tbl[hash_tbl["HASH_N"]=="R"]["%"].sum() if any(hash_tbl["HASH_N"]=="R") else 0
        if left >= 55:
            suggestions.append("Left-hash bias. Offense: set strength to the field and run to space; Defense: set strength to boundary and fit fast to field.")
        if right >= 55:
            suggestions.append("Right-hash bias. Offense: formation to the field, use field-side RPO/screens; Defense: rotate down to wide side.")

    if len(motion_tbl):
        high_motion = motion_tbl[motion_tbl["%"]>=50]
        if len(high_motion):
            suggestions.append("Heavy motion usage. Offense: use motion to ID coverage and leverage; Defense: bump/roll with motion, avoid over-rotating.")

    if not suggestions:
        suggestions.append("Tendencies balanced. Build a call sheet with answers by situation: pressure answers, man beaters, Cover 3/Quarters concepts, and screen menu.")

    return suggestions


# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Controls")
custom_team = st.sidebar.text_input("Team label (optional)", value="Opponent")
show_charts = st.sidebar.checkbox("Show charts", value=True)

st.sidebar.download_button(
    label="Download CSV Template",
    data=template_csv_bytes(),
    file_name="hudl_template.csv",
    mime="text/csv",
)

st.sidebar.markdown("**Universal Personnel**: " + ", ".join(UNIVERSAL_PERSONNEL))
st.sidebar.markdown("**Base Alignments**: " + ", ".join(UNIVERSAL_BASE_ALIGN))

# -----------------------
# File upload
# -----------------------
file = st.file_uploader("Upload Hudl-style CSV", type=["csv"]) 

if not file:
    st.info("Upload a CSV with the headers from the template to begin.")
    st.stop()

# Read & normalize
raw = pd.read_csv(file)
for required in TEMPLATE_COLS:
    if required not in raw.columns:
        raw[required] = np.nan

# Coerce select numeric columns
for col in ["DN","DIST","YARD_LN","QTR"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# Derivations
raw["PLAY_TYPE_NORM"] = raw.apply(normalize_play_type, axis=1)
raw["DIR_LMR"] = raw["PLAY_DIR"].apply(left_mid_right)
raw["DIST_BUCKET"] = raw["DIST"].apply(to_bucket_dist)
raw["HASH_N"] = raw["HASH"].astype(str).str.upper().str[0].map({"L":"L","R":"R","M":"M"}).fillna("U")
raw["RED_ZONE"] = raw["YARD_LN"].apply(red_zone_flag)
raw["FIELD_ZONE"] = raw["YARD_LN"].apply(field_zone)

# Overall tendencies
overall = raw.groupby(["PLAY_TYPE_NORM"]).size().reset_index(name="plays")
if len(overall):
    overall["%"] = (100 * overall["plays"] / overall["plays"].sum()).round(1)

# Grouped tables
by_down = tendency_table(raw, ["DN"])            # DN x PLAY_TYPE
by_dist = tendency_table(raw, ["DN","DIST_BUCKET"])  # DN x DIST_BUCKET x PLAY_TYPE
by_hash = tendency_table(raw, ["HASH_N"])        # HASH x PLAY_TYPE
by_form = tendency_table(raw, ["OFF_FORM","OFF_STR","BACKFIELD"])  # formation x strength x backfield
by_dir = tendency_table(raw, ["DIR_LMR"])        # L/M/R x PLAY_TYPE
by_fz = tendency_table(raw, ["FIELD_ZONE"])      # field zone x PLAY_TYPE
by_rz = tendency_table(raw[raw["RED_ZONE"]=="yes"], ["DN"]) # red zone by down

blitz_3rd = compute_blitz_rate(raw[raw["DN"]==3], ["DIST_BUCKET"])   # 3rd down blitz rate
cov_3rd = compute_coverage(raw[raw["DN"]==3], ["DIST_BUCKET"])       # 3rd down coverage

# Motion usage
motion_tbl = raw.copy()
motion_tbl["MOTION_N"] = np.where(motion_tbl["MOTION"].astype(str).str.strip().eq(""), "None", "Motion")
motion_tbl = motion_tbl.groupby(["MOTION_N","PLAY_TYPE_NORM"]).size().reset_index(name="plays")
den = motion_tbl.groupby(["MOTION_N"])["plays"].transform("sum").replace(0, np.nan)
motion_tbl["%"] = ((100 * motion_tbl["plays"] / den).round(1)).fillna(0)


# -----------------------
# Visuals & Tables
# -----------------------
st.subheader("Overview")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Run/Pass/Screen Mix**")
    st.dataframe(overall)
    if show_charts and len(overall):
        chart_df = overall.set_index("PLAY_TYPE_NORM")["plays"]
        st.bar_chart(chart_df)
with col2:
    st.markdown("**Direction (L/M/R)**")
    st.dataframe(by_dir)

st.divider()

st.subheader("Situational Tendencies")
exp = st.expander("By Down")
with exp:
    st.dataframe(by_down)
    if show_charts and len(by_down):
        pivot = by_down.pivot_table(index="DN", columns="PLAY_TYPE_NORM", values="%", fill_value=0)
        st.bar_chart(pivot)

exp = st.expander("By Down & Distance")
with exp:
    st.dataframe(by_dist)

exp = st.expander("By Formation / Strength / Backfield")
with exp:
    st.dataframe(by_form)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**By Hash**")
    st.dataframe(by_hash)
with c2:
    st.markdown("**By Field Zone**")
    st.dataframe(by_fz)

st.markdown("**Red Zone by Down**")
st.dataframe(by_rz)

st.subheader("3rd Down Study")
cc1, cc2 = st.columns(2)
with cc1:
    st.markdown("**Blitz Rate (3rd Down)**")
    st.dataframe(blitz_3rd)
with cc2:
    st.markdown("**Coverage Usage (3rd Down)**")
    st.dataframe(cov_3rd)

st.subheader("Motion Usage")
st.dataframe(motion_tbl)

# -----------------------
# Suggestions & Call Sheet (Universal)
# -----------------------
suggestions = build_suggestions(overall, by_down, by_dist, by_form, blitz_3rd, cov_3rd, by_hash, motion_tbl)

st.subheader("Auto-Generated Game-Plan Suggestions (Universal)")
for s in suggestions:
    st.markdown(f"- {s}")

# Quick call-sheet blocks (offense)
st.markdown("### Quick Call-Sheet Buckets (Offense — Universal)")
st.markdown("**Vs Pressure (esp. 3rd):** " + ", ".join(OFF_PRESSURE_ANSWERS))
st.markdown("**Vs Man (3rd):** " + ", ".join(OFF_MAN_BEATERS))
st.markdown("**Vs Cover 3 (MOFC):** " + ", ".join(OFF_C3_BEATERS))
st.markdown("**Vs Quarters:** " + ", ".join(OFF_C4_BEATERS))
st.markdown("**Screen Menu:** " + ", ".join(OFF_SCREEN_FAMILY))
st.markdown("**Run Game:** " + ", ".join(RUN_GAME))

# -----------------------
# Exports (CSVs + Markdown)
# -----------------------
outputs = {
    "overall_tendencies.csv": overall,
    "tendency_by_down.csv": by_down,
    "tendency_by_down_distance.csv": by_dist,
    "tendency_by_formation.csv": by_form,
    "tendency_by_direction.csv": by_dir,
    "tendency_by_field_zone.csv": by_fz,
    "tendency_red_zone_by_down.csv": by_rz,
    "blitz_rate_third_down.csv": blitz_3rd,
    "coverage_third_down.csv": cov_3rd,
    "motion_usage.csv": motion_tbl,
}

# Build Markdown report
md_lines = []
md_lines.append(f"# Tendency & Game Plan Report — {custom_team} (Universal)")
md_lines.append("")
md_lines.append("## Overview")
for _, r in overall.sort_values("plays", ascending=False).iterrows():
    md_lines.append(f"- {r['PLAY_TYPE_NORM']}: {int(r['plays'])} plays ({r['%']}%)")
md_lines.append("")
md_lines.append("## Key Tendencies")
md_lines.extend([
    "- By Down: see tendency_by_down.csv",
    "- By Down & Distance: see tendency_by_down_distance.csv",
    "- By Formation/Strength/Backfield: see tendency_by_formation.csv",
    "- Direction (Left/Middle/Right): see tendency_by_direction.csv",
    "- Field Zones & Red Zone: see tendency_by_field_zone.csv and tendency_red_zone_by_down.csv",
    "- Blitz & Coverage on 3rd Down: see blitz_rate_third_down.csv and coverage_third_down.csv",
    "- Motion Usage: see motion_usage.csv",
])
md_lines.append("")
md_lines.append("## Game-Plan Suggestions (Universal)")
for s in suggestions:
    md_lines.append(f"- {s}")
md_lines.append("")
md_lines.append(
    "> Map to call sheet: Pressure answers (screens/quick/hot), Man (mesh/rubs/option/BS fade), Cover 3 (Flood/Curl-Flat/Dagger), Quarters (Posts/Benders/Scissors)."
)
md_text = "\n".join(md_lines)

st.download_button(
    label="⬇️ Download GamePlan_Suggestions.md",
    data=md_text.encode("utf-8"),
    file_name="GamePlan_Suggestions.md",
    mime="text/markdown",
)

# Zip of CSV outputs
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fname, df in outputs.items():
        zf.writestr(fname, df.to_csv(index=False))
    zf.writestr("GamePlan_Suggestions.md", md_text)

st.download_button(
    label="⬇️ Download All Outputs (.zip)",
    data=zip_buf.getvalue(),
    file_name="hudl_tendencies_outputs.zip",
    mime="application/zip",
)

st.success("Done. Universal terminology applied throughout. Adjust your call sheet buckets by situation and personnel.")
