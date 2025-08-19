import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# NEW: persistence & web utils
import base64, json
import urllib.request, urllib.error

# -----------------------
# Hudl Tendency Analyzer (Universal Language)
# Full app with diagnostics, playbook library & Google Sheets sync
# -----------------------

st.set_page_config(page_title="Hudl Tendency Analyzer", layout="wide")
st.title("🏈 Hudl Tendency Analyzer & Game-Plan Builder — Universal Terminology")
st.caption("Upload a Hudl-style CSV (or Excel) to get overall & situational tendencies, diagnostics, and auto-generated game-plan notes using universal football language.")

# -----------------------
# Config & helpers
# -----------------------
PRIMARY_COLS = [
    "PLAY_NUM","ODK","DN","DIST","YARD_LN","HASH","OFF_FORM","OFF_STR","BACKFIELD",
    "OFF_PLAY","PLAY_TYPE","PLAY_DIR","RESULT","GN_LS","EFF","DEF_FRONT","DEF_STR",
    "BLITZ","COVERAGE","QTR","MOTION","MOTION_DIR","PASSER","RECEIVER","PLAY_TYPE_RPS",
    "RUSHER","TEAM"
]

# Alternate header spellings we often see in Hudl exports
ALIAS_MAP = {
    "YARD LN": "YARD_LN",
    "GN/LS": "GN_LS",
    "PLAY #": "PLAY_NUM",
    "PLAY#": "PLAY_NUM",
    "PLAY NO": "PLAY_NUM",
    "PLAY DIR": "PLAY_DIR",
}

DIST_BUCKETS = [
    (0, 3, "short (1-3)"),
    (4, 6, "medium (4-6)"),
    (7, 10, "long (7-10)"),
    (11, 999, "very long (11+)"),
]

# Universal offensive concepts (no team-specific names)
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
OFF_SCREEN_FAMILY = ["RB screen","WR bubble/tunnel","TE screen"]
RUN_GAME = ["Inside Zone","Outside Zone / Stretch","Power","Counter","Iso / Lead","Trap","Pin-pull / Toss"]

@st.cache_data(show_spinner=False)
def template_csv_bytes() -> bytes:
    df = pd.DataFrame(columns=PRIMARY_COLS)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ---- Utilities ----
def to_bucket_dist(dist):
    try:
        d = int(float(dist))
    except Exception:
        return "unknown"
    for lo, hi, name in DIST_BUCKETS:
        if lo <= d <= hi:
            return name
    return "unknown"


def normalize_play_type(row: pd.Series) -> str:
    val = str(row.get("PLAY_TYPE_RPS") or "").strip().upper()
    if val in {"R", "RUN"}: return "Run"
    if val in {"P", "PASS"}: return "Pass"
    if val in {"S", "SCREEN"}: return "Screen"
    t = str(row.get("PLAY_TYPE") or "").lower()
    if "screen" in t: return "Screen"
    if "pass" in t: return "Pass"
    if "run" in t or t in {"ko","po","rush"}: return "Run"
    # Try OFF_PLAY keywords
    op = str(row.get("OFF_PLAY") or "").lower()
    if any(k in op for k in ["screen","bubble","tunnel"]): return "Screen"
    if any(k in op for k in ["pass","boot","play-action","pa "]): return "Pass"
    if any(k in op for k in ["zone","power","counter","iso","trap","toss","draw","run"]): return "Run"
    return "Unknown"


def left_mid_right(s) -> str:
    if pd.isna(s):
        return "Unknown"
    try:
        si = int(float(s))
        return { -1: "Left", 0: "Middle", 1: "Right" }.get(si, "Unknown")
    except Exception:
        pass
    s = str(s).strip().lower()
    if not s: return "Unknown"
    if s in {"l","lt","left"} or s.startswith("l"): return "Left"
    if s in {"r","rt","right"} or s.startswith("r"): return "Right"
    if s in {"m","mid","middle","inside","in","center","ctr"} or s.startswith(("m","i")): return "Middle"
    return "Unknown"


def red_zone_flag(yard_ln: float) -> str:
    if pd.isna(yard_ln): return "no"
    return "yes" if 1 <= yard_ln <= 20 else "no"


def field_zone(yard_ln: float) -> str:
    if pd.isna(yard_ln): return "unknown"
    yl = yard_ln
    if yl <= -90: return "own 10 and in"
    if -89 <= yl <= -21: return "own territory (21-49)"
    if -20 <= yl <= -1: return "near midfield (own 1-20)"
    if 0 <= yl <= 20: return "midfield (50 +/- 20)"
    if 21 <= yl <= 35: return "plus territory (35-21)"
    if 36 <= yl <= 50: return "scoring fringe (20-34)"
    if 1 <= yl <= 20: return "red zone (20 and in)"
    return "unknown"


def tendency_table(df: pd.DataFrame, dims, outcome_col="PLAY_TYPE_NORM"):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=dims + [outcome_col, "plays", "%"])
    g = df.groupby(dims + [outcome_col], dropna=False).size().reset_index(name="plays")
    den = g.groupby(dims)["plays"].transform("sum").replace(0, np.nan)
    g["%"] = ((100 * g["plays"] / den).round(1)).fillna(0)
    return g.sort_values(dims + ["plays"], ascending=[True]*len(dims) + [False])


def compute_blitz_rate(df: pd.DataFrame, dims):
    if len(df) == 0:
        return pd.DataFrame(columns=dims + ["blitz_rate","plays","blitz_rate%"])
    g = df.copy()
    g["BLITZ_FLAG"] = g["BLITZ"].astype(str).str.lower().isin(["1","y","yes","t","true"])
    tbl = g.groupby(dims, dropna=False)["BLITZ_FLAG"].agg(["mean","count"]).reset_index()
    tbl.rename(columns={"mean":"blitz_rate","count":"plays"}, inplace=True)
    tbl["blitz_rate%"] = (100 * tbl["blitz_rate"]).round(1)
    return tbl.sort_values(["plays"], ascending=False)


def compute_coverage(df: pd.DataFrame, dims):
    if len(df) == 0:
        return pd.DataFrame(columns=dims + ["COVERAGE_N","plays","%"])
    g = df.copy()
    g["COVERAGE_N"] = g["COVERAGE"].fillna("Unknown").astype(str).str.upper()
    tbl = g.groupby(dims + ["COVERAGE_N"], dropna=False).size().reset_index(name="plays")
    den = tbl.groupby(dims)["plays"].transform("sum").replace(0, np.nan)
    tbl["%"] = ((100 * tbl["plays"] / den).round(1)).fillna(0)
    return tbl.sort_values(dims + ["plays"], ascending=[True]*len(dims) + [False])


def safe_rate(n, d):
    return (n / d) if d else 0.0

# Success & explosive heuristics from GN_LS (gain/loss)
def coerce_gain(val):
    try:
        return float(val)
    except Exception:
        # sometimes in Hudl it's like "+7" or "-2"
        s = str(val).replace("+", "").strip()
        try:
            return float(s)
        except Exception:
            return np.nan


# ---------- Playbook persistence helpers ----------
PLAYBOOK_COLS = [
    "PLAY_NAME","PERSONNEL","FORMATION","STRENGTH","CONCEPT_TAGS","SITUATION_TAGS","COVERAGE_TAGS","PRESSURE_TAGS","FILE_NAME"
]

@st.cache_data(show_spinner=False)
def play_index_template_bytes() -> bytes:
    df = pd.DataFrame(columns=PLAYBOOK_COLS)
    buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode("utf-8")

# serialize images to base64 so playbook.json can persist them

def serialize_playbook(pb: dict) -> dict:
    out = {
        'plays': pb.get('plays', []),
        'images': {},
        'sheets_csv_url': st.session_state.get('SHEETS_CSV_URL', ''),
        'sheets_write_url': st.session_state.get('SHEETS_WRITE_URL', ''),
    }
    for fname, b in pb.get('images', {}).items():
        try:
            out['images'][fname] = base64.b64encode(b).decode('utf-8')
        except Exception:
            pass
    return out


def deserialize_playbook(pb_json: dict) -> dict:
    pb = {'plays': pb_json.get('plays', []), 'images': {}}
    for fname, s in pb_json.get('images', {}).items():
        try:
            pb['images'][fname] = base64.b64decode(s.encode('utf-8'))
        except Exception:
            pass
    # restore sheet URLs only (no motion settings persisted)
    st.session_state['SHEETS_CSV_URL'] = pb_json.get('sheets_csv_url', '')
    st.session_state['SHEETS_WRITE_URL'] = pb_json.get('sheets_write_url', '')
    return pb

def load_playbook_from_sheets_csv(url: str) -> pd.DataFrame:
    # expects a Google Sheets "Publish to the web" CSV URL
    return pd.read_csv(url)


def push_playbook_to_webhook(url: str, rows: list) -> str:
    # Posts JSON to a Google Apps Script web app (see instructions in chat)
    payload = json.dumps({ 'rows': rows, 'replace': True, 'columns': PLAYBOOK_COLS }).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode('utf-8')


# ---------- Suggestions builder ----------

def build_suggestions(overall, by_down, by_dist, by_form, blitz_3rd, cov_3rd, hash_tbl, motion_tbl, sample_size, sr_overall, xpl_overall):
    suggestions = []

    # Sample size guardrails
    if sample_size < 25:
        suggestions.append(f"Warning: small sample ({sample_size} plays). Treat tendencies with caution.")

    # Overall run/pass/screen
    total = overall["plays"].sum() if len(overall) else 0
    rp = overall[overall["PLAY_TYPE_NORM"].isin(["Run","Pass","Screen"])]
    run_p = safe_rate(rp[rp["PLAY_TYPE_NORM"]=="Run"]["plays"].sum(), total)
    pass_p = safe_rate(rp[rp["PLAY_TYPE_NORM"]=="Pass"]["plays"].sum(), total)
    screen_p = safe_rate(rp[rp["PLAY_TYPE_NORM"]=="Screen"]["plays"].sum(), total)

    if run_p >= 0.60:
        suggestions.append(f"Run-heavy profile overall ({run_p:.0%}). Def: extra fitter on early downs; Off: play-action shots.")
    elif pass_p >= 0.60:
        suggestions.append(f"Pass-heavy profile overall ({pass_p:.0%}). Def: simulated pressures, late rotation from two-high.")
    if screen_p >= 0.10:
        suggestions.append(f"Screens appear {screen_p:.0%}. DL retrace; keep RB/WR/TE screen menu ready.")

    # Efficiency context
    if not np.isnan(sr_overall):
        suggestions.append(f"Estimated success rate: {sr_overall:.0%} (success = gain ≥ yard-to-go on 1st/2nd, or conversion on 3rd).")
    if not np.isnan(xpl_overall):
        suggestions.append(f"Explosive rate: {xpl_overall:.0%} (10+ rush / 15+ pass).")

    # Down tendencies
    for d in [1,2,3]:
        sub = by_down[by_down["DN"]==d]
        if len(sub):
            r = sub[sub["PLAY_TYPE_NORM"]=="Run"]["%"].sum() if "Run" in set(sub["PLAY_TYPE_NORM"]) else 0
            p = sub[sub["PLAY_TYPE_NORM"]=="Pass"]["%"].sum() if "Pass" in set(sub["PLAY_TYPE_NORM"]) else 0
            if d==1 and r>=60:
                suggestions.append("1st down leans run. Fit downhill; be alert for play-action off base runs.")
            if d==3 and p>=70:
                suggestions.append("3rd down leans pass. Def: simulated pressure, play the sticks. Off: quick man/zone beaters.")

    # Distance tendencies (3rd down)
    td = by_dist[(by_dist["DN"]==3)]
    if len(td):
        long_pass = td[(td["DIST_BUCKET"]=="long (7-10)") & (td["PLAY_TYPE_NORM"]=="Pass")]
        if len(long_pass) and long_pass["%"].iloc[0] >= 70:
            suggestions.append("3rd & 7–10 high pass tendency. Off: Flood/Sail or Dagger. Def: mug A-gaps, spin to 3-robber.")
        vl_pass = td[(td["DIST_BUCKET"]=="very long (11+)") & (td["PLAY_TYPE_NORM"]=="Pass")]
        if len(vl_pass) and vl_pass["%"].iloc[0] >= 80:
            suggestions.append("3rd & 11+ = must-pass. Spy QB if mobile; Off: max-protect shots vs soft zone.")

    # By formation tendencies (top formation)
    if len(by_form):
        top_form = by_form.groupby(["OFF_FORM"])["plays"].sum().reset_index().sort_values("plays", ascending=False).head(1)
        if len(top_form):
            f = str(top_form["OFF_FORM"].iloc[0])
            f_tbl = by_form[by_form["OFF_FORM"]==f]
            run_bias = f_tbl[f_tbl["PLAY_TYPE_NORM"]=="Run"]["%"].sum() if "Run" in set(f_tbl["PLAY_TYPE_NORM"]) else 0
            if run_bias >= 65:
                suggestions.append(f"In {f}, strong run tendency (~{run_bias:.0f}%). Def: set edges, close interior gaps. Off: play-action counters from same look.")

    # Blitz on 3rd down
    if len(blitz_3rd) and (blitz_3rd["blitz_rate"] >= 0.35).any():
        suggestions.append("They pressure on 3rd (≥35%). Off: screens, quick game, hot throws; consider max-protect shots.")

    # Coverage on 3rd down
    if len(cov_3rd):
        top_cov = cov_3rd.groupby(["COVERAGE_N"])["plays"].sum().reset_index().sort_values("plays", ascending=False).head(1)
        if len(top_cov):
            cov = top_cov["COVERAGE_N"].iloc[0]
            if cov in {"COVER 1","C1","MAN","COVER1"}:
                suggestions.append("3rd = Man. Off: Mesh, rub/stack, option routes, back-shoulder.")
            if cov in {"COVER 3","C3","THREE","COVER3"}:
                suggestions.append("3rd = Cover 3 (MOFC). Off: Flood/Sail, Curl-Flat, Dagger; attack seams/curl window.")
            if cov in {"COVER 4","C4","QUARTERS"}:
                suggestions.append("3rd = Quarters. Off: posts & benders, scissors, deep overs.")

    # Hash tendencies
    if len(hash_tbl):
        left = hash_tbl[hash_tbl["HASH_N"]=="L"]["%"].sum() if any(hash_tbl["HASH_N"]=="L") else 0
        right = hash_tbl[hash_tbl["HASH_N"]=="R"]["%"].sum() if any(hash_tbl["HASH_N"]=="R") else 0
        if left >= 55:
            suggestions.append("Left-hash bias. Off: set strength to field; Def: set strength to boundary, fit fast to field.")
        if right >= 55:
            suggestions.append("Right-hash bias. Off: formation into field with RPO/screens; Def: rotate down to wide side.")

    # Motion tendencies
    if len(motion_tbl) and (motion_tbl["%"] >= 50).any():
        suggestions.append("Heavy motion usage. Off: use motion to ID coverage/leverage; Def: bump/roll with motion.")

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

# -----------------------
# Playbook Library — Upload once, reuse forever (with Google Sheets sync)
# (Moved ABOVE data upload so you can set up your playbook anytime)
# -----------------------

st.subheader("Playbook Library — Upload once, reuse forever")

# Session defaults
if 'PLAYBOOK' not in st.session_state:
    st.session_state.PLAYBOOK = { 'plays': [], 'images': {} }
if 'SHEETS_CSV_URL' not in st.session_state:
    st.session_state.SHEETS_CSV_URL = ''
if 'SHEETS_WRITE_URL' not in st.session_state:
    st.session_state.SHEETS_WRITE_URL = ''
# --- Formation & screen defaults ---
if 'CONCEPT_FORMATION_RULES' not in st.session_state:
    # Default based on your note: TRAIN only out of TRIPS (not BUNCH)
    st.session_state.CONCEPT_FORMATION_RULES = {
        'TRAIN': ['TRIPS'],
        # Add more here as you decide, e.g.: 'VIPER': ['TRIPS','TWIG']
    }
if 'SCREEN_RECIPIENT_ORDER' not in st.session_state:
    st.session_state.SCREEN_RECIPIENT_ORDER = ['Y','Z','H','X','F']

col_pb1, col_pb2 = st.columns([2,1])
with col_pb1:
    st.markdown("**A) Load or build your library**")
    playbook_json_file = st.file_uploader("Load playbook.json (optional)", type=["json"], key="pbjson")
    if playbook_json_file:
        try:
            pb = json.loads(playbook_json_file.read().decode('utf-8'))
            st.session_state.PLAYBOOK = deserialize_playbook(pb)
            st.success(f"Loaded {len(st.session_state.PLAYBOOK.get('plays', []))} plays from your library.")
        except Exception as e:
            st.error(f"Couldn't load playbook.json: {e}")
    st.markdown("**Upload play images** (PNG/JPG/WEBP or ZIP). Name files to match FILE_NAME in your index, or exactly the PLAY_NAME.")
    uploads = st.file_uploader(
        "Add play screenshots/diagrams (PNG/JPG/WEBP or .zip)",
        type=["png","jpg","jpeg","webp","zip"],
        accept_multiple_files=True,
        key="pbimgs",
    )
    if uploads:
        added, added_from_zip, skipped = 0, 0, 0
        for f in uploads:
            lower = f.name.lower()
            if lower.endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
                        for zi in zf.infolist():
                            if zi.is_dir():
                                continue
                            ext = Path(zi.filename).suffix.lower()
                            if ext not in {".png",".jpg",".jpeg",".webp"}:
                                skipped += 1
                                continue
                            try:
                                b = zf.read(zi.filename)
                                base = Path(zi.filename).name
                                st.session_state.PLAYBOOK['images'][base] = b
                                added += 1
                                added_from_zip += 1
                            except Exception:
                                skipped += 1
                except Exception as e:
                    st.error(f"Couldn't read zip '{f.name}': {e}")
            else:
                st.session_state.PLAYBOOK['images'][f.name] = f.read()
                added += 1
        msg = f"Stored {added} image(s)"
        if added_from_zip:
            msg += f" ({added_from_zip} from zip)"
        if skipped:
            msg += f"; skipped {skipped} non-image file(s)"
        msg += " in your library (in-memory)."
        st.success(msg)

    st.markdown("**Index your plays with a CSV** (or add rows manually).")
