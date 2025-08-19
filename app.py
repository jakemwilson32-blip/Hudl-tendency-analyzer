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

    st.markdown("**Upload play images** (PNG/JPG). Name files to match FILE_NAME in your index.")
    img_files = st.file_uploader("Add play screenshots/diagrams", type=["png","jpg","jpeg"], accept_multiple_files=True, key="pbimgs")
    if img_files:
        for f in img_files:
            st.session_state.PLAYBOOK['images'][f.name] = f.read()
        st.success(f"Stored {len(img_files)} image(s) in your library (in-memory).")

    st.markdown("**Index your plays with a CSV** (or add rows manually).")
    st.download_button("Download Play Index CSV Template", data=play_index_template_bytes(), file_name="play_index_template.csv", mime="text/csv")
    play_index_csv = st.file_uploader("Upload play_index.csv (columns: " + ", ".join(PLAYBOOK_COLS) + ")", type=["csv"], key="pbidx")
    if play_index_csv:
        try:
            df_idx = pd.read_csv(play_index_csv)
            missing_cols = [c for c in PLAYBOOK_COLS if c not in df_idx.columns]
            if missing_cols:
                st.error("Missing columns in play_index.csv: " + ", ".join(missing_cols))
            else:
                rows = df_idx.to_dict(orient='records')
                st.session_state.PLAYBOOK['plays'] = rows
                st.success(f"Indexed {len(rows)} plays.")
        except Exception as e:
            st.error(f"Could not read play_index.csv: {e}")

    st.markdown("**Save your playbook** (download a .json to reuse later)")
    pb_bytes = json.dumps(serialize_playbook(st.session_state.PLAYBOOK)).encode('utf-8')
    st.download_button("⬇️ Save Playbook Library (playbook.json)", data=pb_bytes, file_name="playbook.json", mime="application/json")

with col_pb2:
    st.markdown("**B) Google Sheets Sync (optional)**")
    st.caption("No installs. Use a Google Sheet as your play index. 'Publish to web' → paste CSV link below.")
    sheets_url = st.text_input("Google Sheets — Published CSV URL (read-only)", value=st.session_state.get('SHEETS_CSV_URL',''), placeholder="https://docs.google.com/spreadsheets/d/e/.../pub?gid=0&single=true&output=csv")
    if st.button("Load from Google Sheets"):
        try:
            df_idx = load_playbook_from_sheets_csv(sheets_url)
            missing_cols = [c for c in PLAYBOOK_COLS if c not in df_idx.columns]
            if missing_cols:
                st.error("Your sheet is missing columns: " + ", ".join(missing_cols))
            else:
                st.session_state.PLAYBOOK['plays'] = df_idx.to_dict(orient='records')
                st.session_state.SHEETS_CSV_URL = sheets_url
                st.success(f"Loaded {len(df_idx)} plays from Google Sheets.")
        except Exception as e:
            st.error(f"Couldn't load Google Sheets CSV: {e}")

    writer_url = st.text_input("Google Apps Script Web App URL (for Save → Sheets, optional)", value=st.session_state.get('SHEETS_WRITE_URL',''), placeholder="https://script.google.com/macros/s/AKfycb.../exec")
    if st.button("Push library to Google Sheets"):
        try:
            resp = push_playbook_to_webhook(writer_url, st.session_state.PLAYBOOK.get('plays', []))
            st.session_state.SHEETS_WRITE_URL = writer_url
            st.success("Pushed to Google Sheets. Response: " + str(resp)[:200])
        except Exception as e:
            st.error(f"Push failed: {e}")

    st.markdown("**Quick Add (single play)**")
    with st.form("quick_add_play"):
        qa_name = st.text_input("Play Name")
        qa_personnel = st.text_input("Personnel (e.g., 11, 12, 20)")
        qa_form = st.text_input("Formation (e.g., Trips, Bunch, Singleback)")
        qa_strength = st.text_input("Strength (Right/Left/Boundary/Field)")
        qa_concepts = st.text_input("Concept tags (comma-separated: Flood, Mesh, Curl-Flat, PA, Boot, RPO, IZ, OZ, Power, Counter)")
        qa_situ = st.text_input("Situation tags (3rd&7-10, 1st&10, RZ, 2-minute, vs Nickel)")
        qa_cov = st.text_input("Coverage tags (vs Man, vs C3, vs Quarters)")
        qa_press = st.text_input("Pressure tags (vs Blitz, vs 5-man, vs DB blitz)")
        qa_file = st.text_input("Image file name (must match an uploaded image)")
        submitted = st.form_submit_button("Add Play")
        if submitted:
            st.session_state.PLAYBOOK['plays'].append({
                'PLAY_NAME': qa_name,
                'PERSONNEL': qa_personnel,
                'FORMATION': qa_form,
                'STRENGTH': qa_strength,
                'CONCEPT_TAGS': qa_concepts,
                'SITUATION_TAGS': qa_situ,
                'COVERAGE_TAGS': qa_cov,
                'PRESSURE_TAGS': qa_press,
                'FILE_NAME': qa_file,
            })
            st.success(f"Added play: {qa_name}")

# -----------------------
# Call Sheet Rules (formations & screens)
# -----------------------
with st.expander("Call Sheet Rules (formations & screens)"):
    # Formation constraints per concept (e.g., TRAIN only from TRIPS)
    lib_for_opts = pd.DataFrame(st.session_state.PLAYBOOK.get('plays', []))
    form_opts = sorted(set(str(x).upper() for x in lib_for_opts.get('FORMATION', pd.Series(dtype=str)).dropna().unique())) or ["DICE","DECK","DOUBLES","DOS","TRIPS","TWIG","TREY","TRIO","BUNCH","BUNDLE","GANG","GLOCK","GAUGE","EMPTY","EGO"]
    rules = st.session_state.get('CONCEPT_FORMATION_RULES', {})
    def _rule_ui(concept_key: str, label: str):
        current = [s.upper() for s in rules.get(concept_key, [])]
        picked = st.multiselect(f"Allowed formations for {label}", options=form_opts, default=current, key=f"rule_{concept_key}")
        rules[concept_key] = picked
    _rule_ui('TRAIN','TRAIN (tunnel screen)')
    _rule_ui('VIPER','VIPER (swing screen)')
    _rule_ui('UNICORN','UNICORN (throwback)')
    _rule_ui('UTAH','UTAH (shovel/drag)')
    st.session_state.CONCEPT_FORMATION_RULES = rules

    st.markdown("**Screen recipient letters (order of preference)**")
    order_str = st.text_input("Letters in order (comma-separated)", value=",".join(st.session_state.get('SCREEN_RECIPIENT_ORDER', ['Y','Z','H','X','F'])))
    try:
        parsed = [s.strip().upper() for s in order_str.split(',') if s.strip()]
        if parsed:
            st.session_state.SCREEN_RECIPIENT_ORDER = parsed
    except Exception:
        pass

# -----------------------
# File upload (optional)
# -----------------------
file = st.file_uploader("Upload Hudl-style CSV or Excel (optional)", type=["csv","xlsx"]) 

# If no file, proceed with an empty DataFrame so the app can still build a plan from your Playbook Library
if not file:
    st.info("No opponent data uploaded — that's okay! We'll still build a game plan from your Playbook Library below.")
    raw = pd.DataFrame(columns=PRIMARY_COLS)
else:
    # Try to read the file; prefer CSV; Excel requires openpyxl on the server
    try:
        if file.name.lower().endswith(".xlsx"):
            raw = pd.read_excel(file, engine="openpyxl")
        else:
            raw = pd.read_csv(file)
    except ImportError:
        st.warning("Excel support requires 'openpyxl' on the server. Either upload a CSV or add 'openpyxl' to requirements.txt. Proceeding without opponent data.")
        raw = pd.DataFrame(columns=PRIMARY_COLS)

# Rename aliases to primary
raw = raw.rename(columns={k:v for k,v in ALIAS_MAP.items() if k in raw.columns})

# Ensure all primary columns exist
for c in PRIMARY_COLS:
    if c not in raw.columns:
        raw[c] = np.nan

# Coerce key numerics
for col in ["DN","DIST","YARD_LN","QTR","GN_LS"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# Derivations
raw["PLAY_TYPE_NORM"] = raw.apply(normalize_play_type, axis=1)
raw["DIR_LMR"] = raw["PLAY_DIR"].apply(left_mid_right)
raw["DIST_BUCKET"] = raw["DIST"].apply(to_bucket_dist)
raw["HASH_N"] = raw["HASH"].astype(str).str.upper().str[0].map({"L":"L","R":"R","M":"M"}).fillna("U")
raw["RED_ZONE"] = raw["YARD_LN"].apply(red_zone_flag)
raw["FIELD_ZONE"] = raw["YARD_LN"].apply(field_zone)

# Derive gains if provided in alias column
if "GN/LS" in raw.columns and raw["GN_LS"].isna().all():
    raw["GN_LS"] = raw["GN/LS"].apply(coerce_gain)

# Success rate heuristic (needs DN, DIST, GN_LS)
# success: on 1st, gain >= 0.5*to-go; on 2nd, gain >= 0.7*to-go; on 3rd/4th, gain >= to-go
def success_row(r):
    dn, dist, g = r.get("DN"), r.get("DIST"), r.get("GN_LS")
    if pd.isna(dn) or pd.isna(dist) or pd.isna(g):
        return np.nan
    try:
        dn = int(dn); dist = float(dist); g = float(g)
    except Exception:
        return np.nan
    if dn == 1:
        return g >= 0.5*dist
    if dn == 2:
        return g >= 0.7*dist
    return g >= dist

raw["SUCCESS"] = raw.apply(success_row, axis=1)

# Explosive rate heuristic (10+ rush, 15+ pass)
raw["EXPLOSIVE"] = False
raw.loc[(raw["PLAY_TYPE_NORM"]=="Run") & (raw["GN_LS"] >= 10), "EXPLOSIVE"] = True
raw.loc[(raw["PLAY_TYPE_NORM"]=="Pass") & (raw["GN_LS"] >= 15), "EXPLOSIVE"] = True

# --- Diagnostics ---
st.subheader("Data Diagnostics")
key_cols = ["DN","DIST","YARD_LN","HASH","PLAY_TYPE","PLAY_TYPE_RPS","OFF_FORM","OFF_STR","BACKFIELD","PLAY_DIR","BLITZ","COVERAGE","MOTION","GN_LS"]
missing = [c for c in key_cols if raw[c].isna().all()]
non_null_counts = raw[key_cols].notna().sum().to_frame("non-null").T
st.write("**Non-null counts (how much data do we have by column):**")
st.dataframe(non_null_counts)
if missing:
    st.warning("No data found in: " + ", ".join(missing))

# --- Overall tendencies ---
overall = raw.groupby(["PLAY_TYPE_NORM"]).size().reset_index(name="plays")
if len(overall):
    overall["%"] = (100 * overall["plays"] / overall["plays"].sum()).round(1)

sample_size = int(overall["plays"].sum()) if len(overall) else int(len(raw))
sr_overall = raw["SUCCESS"].mean() if raw["SUCCESS"].notna().any() else np.nan
xpl_overall = raw["EXPLOSIVE"].mean() if raw["GN_LS"].notna().any() else np.nan

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
motion_tbl = motion_tbl.groupby(["MOTION_N","PLAY_TYPE_NORM"], dropna=False).size().reset_index(name="plays")
den = motion_tbl.groupby(["MOTION_N"])['plays'].transform('sum').replace(0, np.nan)
motion_tbl["%"] = ((100 * motion_tbl["plays"] / den).round(1)).fillna(0)

# -----------------------
# Visuals & Tables
# -----------------------
st.subheader("Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Run/Pass/Screen Mix**")
    if len(overall):
        st.dataframe(overall)
        if show_charts:
            st.bar_chart(overall.set_index("PLAY_TYPE_NORM")["plays"])
    else:
        st.info("No PLAY_TYPE found. Check columns PLAY_TYPE or PLAY_TYPE_RPS or OFF_PLAY keywords.")
with col2:
    st.markdown("**Direction (L/M/R)**")
    if len(by_dir): st.dataframe(by_dir)
    else: st.info("No direction data (PLAY_DIR) present.")
with col3:
    st.markdown("**Efficiency**")
    eff_lines = []
    if not np.isnan(sr_overall): eff_lines.append(f"Success rate: {sr_overall:.0%}")
    if not np.isnan(xpl_overall): eff_lines.append(f"Explosive rate: {xpl_overall:.0%}")
    if eff_lines:
        st.write("\n".join(eff_lines))
    else:
        st.info("Add GN_LS (gain/loss) to compute success & explosive rates.")

st.divider()

st.subheader("Situational Tendencies")
exp = st.expander("By Down", expanded=True)
with exp:
    if len(by_down):
        st.dataframe(by_down)
        if show_charts:
            pivot = by_down.pivot_table(index="DN", columns="PLAY_TYPE_NORM", values="%", fill_value=0)
            st.bar_chart(pivot)
    else:
        st.info("Missing DN or PLAY_TYPE.")

exp = st.expander("By Down & Distance")
with exp:
    if len(by_dist): st.dataframe(by_dist)
    else: st.info("Missing DIST (yards to go) or PLAY_TYPE.")

exp = st.expander("By Formation / Strength / Backfield")
with exp:
    if len(by_form): st.dataframe(by_form)
    else: st.info("Missing OFF_FORM / OFF_STR / BACKFIELD.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**By Hash**")
    if len(by_hash): st.dataframe(by_hash)
    else: st.info("Missing HASH.")
with c2:
    st.markdown("**By Field Zone**")
    if len(by_fz): st.dataframe(by_fz)
    else: st.info("Missing YARD_LN.")

st.markdown("**Red Zone by Down**")
if len(by_rz): st.dataframe(by_rz)
else: st.info("No red-zone snaps detected (YARD_LN 1–20).")

st.subheader("3rd Down Study")
cc1, cc2 = st.columns(2)
with cc1:
    st.markdown("**Blitz Rate (3rd Down)**")
    if len(blitz_3rd): st.dataframe(blitz_3rd)
    else: st.info("Need BLITZ flags on 3rd down to compute.")
with cc2:
    st.markdown("**Coverage Usage (3rd Down)**")
    if len(cov_3rd): st.dataframe(cov_3rd)
    else: st.info("Need COVERAGE values on 3rd down to compute.")

st.subheader("Motion Usage")
if len(motion_tbl): st.dataframe(motion_tbl)
else: st.info("No motion data found.")

# -----------------------
# Suggestions & Call Sheet (Universal)
# -----------------------
suggestions = build_suggestions(overall, by_down, by_dist, by_form, blitz_3rd, cov_3rd, by_hash, motion_tbl, sample_size, sr_overall, xpl_overall)

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
# Playbook-Only Planner (No Opponent Data Required)
# -----------------------
st.subheader("Playbook-Only Planner (No Opponent Data)")
lib = pd.DataFrame(st.session_state.PLAYBOOK.get('plays', []))
if lib.empty:
    st.info("Add plays to your Playbook Library above to generate a call sheet without opponent data.")
else:
    def _text_contains(text: str, keywords: list[str]) -> bool:
        hay = str(text or '').lower()
        return any(k in hay for k in keywords)

    def _pick_varied(df: pd.DataFrame, keywords: list[str], limit: int = 6) -> pd.DataFrame:
        mask = df.apply(lambda r: _text_contains((r.get('CONCEPT_TAGS','') + ' ' + r.get('SITUATION_TAGS','')), keywords), axis=1)
        cand = df[mask].copy()
        if cand.empty:
            cand = df.copy()
        picked = []
        used_names, used_forms = set(), set()
        for _, r in cand.iterrows():
            name = r.get('PLAY_NAME'); form = r.get('FORMATION')
            if name in used_names or form in used_forms:
                continue
            picked.append(r)
            used_names.add(name); used_forms.add(form)
            if len(picked) >= limit:
                break
        return pd.DataFrame(picked)

    buckets = {
        "1st & 10": ["1st&10","quick","inside zone","iz","outside zone","oz","power","counter","rpo","play-action","boot"],
        "2nd & medium (4-6)": ["2nd&medium","quick","screen","draw","counter","boot","pa"],
        "3rd & short (1-3)": ["3rd&1-3","hitch","slant","snag","mesh","iso","power","qb sneak"],
        "3rd & medium (4-6)": ["3rd&4-6","snag","smash","flood","curl-flat","mesh","screen"],
        "3rd & long (7-10)": ["3rd&7-10","dagger","flood","smash","screen","wheel"],
        "Red Zone": ["rz","money","atm","smash","snag","fade","rub","pick"],
        "2-minute": ["2-minute","hurry","sideline","dagger","curl-flat","out","arrow"],
    }

    # Apply formation constraints per concept before selection
    def _detect_concept(row: pd.Series) -> str | None:
        txt = (str(row.get('CONCEPT_TAGS','')) + ' ' + str(row.get('PLAY_NAME',''))).upper()
        for key in ['TRAIN','VIPER','UNICORN','UTAH']:
            if key in txt:
                return key
        return None

    rules = st.session_state.get('CONCEPT_FORMATION_RULES', {})
    lib2 = lib.copy()
    if not lib2.empty:
        lib2['__CONCEPT__'] = lib2.apply(_detect_concept, axis=1)
        mask_keep = []
        for _, rr in lib2.iterrows():
            ck = rr['__CONCEPT__']
            if ck and rules.get(ck):
                allowed = {s.upper() for s in rules[ck]}
                form = str(rr.get('FORMATION','')).upper()
                mask_keep.append(form in allowed)
            else:
                mask_keep.append(True)
        lib2 = lib2[pd.Series(mask_keep, index=lib2.index)].drop(columns=['__CONCEPT__'])

    call_rows = []
    for bucket, keys in buckets.items():
        picked = _pick_varied(lib2, [k.lower() for k in keys], limit=6)
        if not picked.empty:
            picked = picked[["PLAY_NAME","PERSONNEL","FORMATION","STRENGTH","CONCEPT_TAGS","SITUATION_TAGS","COVERAGE_TAGS","PRESSURE_TAGS","FILE_NAME"]]
            picked.insert(0, "Bucket", bucket)
            call_rows.append(picked)
    if call_rows:
        call_sheet = pd.concat(call_rows, ignore_index=True)

        # Detect screen-like concepts
        def _screen_concept(row: pd.Series) -> str | None:
            txt = (str(row.get('CONCEPT_TAGS','')) + ' ' + str(row.get('PLAY_NAME',''))).upper()
            for key in ['TRAIN','VIPER','UNICORN','UTAH']:
                if key in txt:
                    return key
            if 'SCREEN' in txt:
                return 'SCREEN'
            return None

        # Recipient order for screens (no motions)
        order = st.session_state.get('SCREEN_RECIPIENT_ORDER', ['Y','Z','H','X','F'])

        call_sheet = call_sheet.copy()
        call_sheet["RECIPIENT_LETTER"] = ""

        # Base CALL (no motions): "<Formation> <Strength>. <PLAY_NAME>"
        base_prefix = (
            call_sheet["FORMATION"].fillna("").str.strip() + " " +
            call_sheet["STRENGTH"].fillna("").str.strip()
        ).str.strip()
        call_sheet["CALL"] = base_prefix.where(base_prefix.eq(""), base_prefix + ". ") + call_sheet["PLAY_NAME"].fillna("")

        # For screens, override CALL with recipient-coded version
        screen_idx = [i for i in call_sheet.index if _screen_concept(call_sheet.loc[i])]
        for j, idx in enumerate(screen_idx):
            recip = order[j % len(order)]
            concept = _screen_concept(call_sheet.loc[idx]) or 'SCREEN'
            form = str(call_sheet.at[idx, 'FORMATION'] or '').strip()
            strength = str(call_sheet.at[idx, 'STRENGTH'] or '').strip()
            prefix = (form + (' ' + strength if strength else '')).strip()
            call_sheet.at[idx, 'RECIPIENT_LETTER'] = recip
            call_sheet.at[idx, 'CALL'] = (f"{prefix}. " if prefix else '') + f"{recip.lower()}-{concept.lower()}"

        cols = ["Bucket","CALL","PLAY_NAME","RECIPIENT_LETTER",
                "PERSONNEL","FORMATION","STRENGTH","CONCEPT_TAGS","SITUATION_TAGS",
                "COVERAGE_TAGS","PRESSURE_TAGS","FILE_NAME"]
        call_sheet = call_sheet[[c for c in cols if c in call_sheet.columns]]

        st.dataframe(call_sheet)
        st.download_button("⬇️ Download Call_Sheet_PlaybookOnly.csv", data=call_sheet.to_csv(index=False).encode("utf-8"), file_name="Call_Sheet_PlaybookOnly.csv", mime="text/csv")
    else:
        st.info("Could not find plays matching the standard buckets. Add situation tags like '1st&10', '3rd&7-10', or concept tags (Snag, Flood, Mesh, Smash, Dagger, Screen, IZ/OZ/Power/Counter).")

# -----------------------
# Offense-Focused Matchup Builder (Our O vs Their D)
# -----------------------
with st.expander("Offense-Focused Matchup Builder (Our O vs Their D)", expanded=False):
    st.caption("Upload defensive data for the opponent and (optionally) our offensive data. We'll rank your Playbook Library plays by matchup fit: our best concepts vs their most-used coverages & pressure.")
    opp_file = st.file_uploader("Opponent DEF Hudl CSV/Excel", type=["csv","xlsx"], key="oppdef")
    off_file = st.file_uploader("Our OFF Hudl CSV/Excel (optional)", type=["csv","xlsx"], key="ouroff")

    def _load_file(f):
        if not f: return None
        try:
            return pd.read_excel(f, engine="openpyxl") if f.name.lower().endswith(".xlsx") else pd.read_csv(f)
        except Exception:
            try: return pd.read_csv(f)
            except Exception: return None

    opp_df = _prep_hudl(_load_file(opp_file))
    off_df = _prep_hudl(_load_file(off_file))

    call_sheet_matchup = None
    if not lib.empty and not opp_df.empty:
        # --- Opponent coverage & pressure profile ---
        def _cov_norm(s: str) -> str:
            s = str(s or '').upper()
            if any(k in s for k in ["COVER 1", "C1", "MAN"]): return "MAN"
            if any(k in s for k in ["COVER 3", "C3", "THREE"]): return "C3"
            if any(k in s for k in ["COVER 4", "C4", "QUARTERS"]): return "C4"
            if any(k in s for k in ["COVER 2", "C2", "TWO"]): return "C2"
            return "UNK"

opp_cov = opp_df.copy()
# robust to missing COVERAGE column
cov_series = opp_cov.get("COVERAGE", pd.Series(index=opp_cov.index, dtype=object))
opp_cov["COVN"] = cov_series.astype(str).apply(_cov_norm)
cov_dist = opp_cov["COVN"].value_counts(normalize=True).to_dict()

blitz_3rd_tbl = compute_blitz_rate(opp_df[opp_df["DN"] == 3], ["DIST_BUCKET"]) if "DIST_BUCKET" in opp_df.columns and "DN" in opp_df.columns else pd.DataFrame()
        opp_blitz3 = float(blitz_3rd_tbl["blitz_rate"].mean()) if len(blitz_3rd_tbl) else 0.0

        # --- Our concept success profile (optional) ---
        concept_map = {
            "mesh":["mesh"], "smash":["smash"], "flood":["flood","sail"], "dagger":["dagger"],
            "curlflat":["curl flat","curl-flat","curl","flat"], "screen":["screen","bubble","tunnel"],
            "iz":["inside zone","iz"], "oz":["outside zone","oz","stretch"], "power":["power"], "counter":["counter"],
            "wheel":["wheel"], "hitch":["hitch"], "arrow":["arrow"], "stick":["stick","snag"],
        }
        def _infer_concepts(txt: str) -> list[str]:
            t = str(txt or '').lower()
            hits = [k for k, kws in concept_map.items() if any(kw in t for kw in kws)]
            return hits or ["other"]

        if off_df is not None and not off_df.empty:
            txtcol = off_df.get("OFF_PLAY") if "OFF_PLAY" in off_df.columns else off_df.get("PLAY_TYPE", pd.Series(dtype=str))
            tmp = pd.DataFrame({"concepts": txtcol.apply(_infer_concepts), "SUCCESS": off_df.get("SUCCESS")})
            tmp = tmp.explode("concepts")
            concept_success = tmp.groupby("concepts")["SUCCESS"].mean().to_dict()
        else:
            concept_success = {}

        # --- Score each library play by matchup fit ---
        def _score_row(r: pd.Series) -> float:
            score = 0.0
            cov_tags = str(r.get('COVERAGE_TAGS','')).lower()
            if 'vs man' in cov_tags: score += 1.2 * cov_dist.get('MAN', 0)
            if 'vs c3' in cov_tags or 'vs cover 3' in cov_tags: score += 1.2 * cov_dist.get('C3', 0)
            if 'vs quarters' in cov_tags or 'vs c4' in cov_tags or 'vs cover 4' in cov_tags: score += 1.2 * cov_dist.get('C4', 0)
            if 'vs cover 2' in cov_tags or 'vs c2' in cov_tags: score += 1.2 * cov_dist.get('C2', 0)
            if opp_blitz3 >= 0.35 and 'vs blitz' in str(r.get('PRESSURE_TAGS','')).lower(): score += 0.4

            cons = str(r.get('CONCEPT_TAGS','')).lower()
            hits = [k for k, kws in concept_map.items() if any(w in cons for w in kws)]
            if hits:
                score += float(np.mean([concept_success.get(k, 0.5) for k in hits]))  # default neutral 0.5
            return float(score)

        lib_scored = lib.copy()
        lib_scored["__SCORE__"] = lib_scored.apply(_score_row, axis=1)

        def _pick_top(df: pd.DataFrame, keywords: list[str], limit: int = 6) -> pd.DataFrame:
            def _contains(row):
                hay = (str(row.get('CONCEPT_TAGS','')) + ' ' + str(row.get('SITUATION_TAGS','')) + ' ' + str(row.get('PLAY_NAME',''))).lower()
                return any(k in hay for k in keywords)
            cand = df[df.apply(_contains, axis=1)].copy()
            if cand.empty:
                cand = df.copy()
            cand = cand.sort_values("__SCORE__", ascending=False)
            picked, used_names, used_forms = [], set(), set()
            for _, rr in cand.iterrows():
                name, form = rr.get('PLAY_NAME'), rr.get('FORMATION')
                if name in used_names or form in used_forms:
                    continue
                picked.append(rr); used_names.add(name); used_forms.add(form)
                if len(picked) >= limit:
                    break
            return pd.DataFrame(picked)

        call_rows2 = []
        for bucket, keys in {
            "1st & 10": ["1st&10","quick","inside zone","iz","outside zone","oz","power","counter","rpo","play-action","boot"],
            "2nd & medium (4-6)": ["2nd&medium","quick","screen","draw","counter","boot","pa"],
            "3rd & short (1-3)": ["3rd&1-3","hitch","slant","snag","mesh","iso","power","qb sneak"],
            "3rd & medium (4-6)": ["3rd&4-6","snag","smash","flood","curl-flat","mesh","screen"],
            "3rd & long (7-10)": ["3rd&7-10","dagger","flood","smash","screen","wheel"],
            "Red Zone": ["rz","money","atm","smash","snag","fade","rub","pick"],
            "2-minute": ["2-minute","hurry","sideline","dagger","curl-flat","out","arrow"],
        }.items():
            got = _pick_top(lib_scored, [k.lower() for k in keys], limit=6)
            if not got.empty:
                got = got[["PLAY_NAME","PERSONNEL","FORMATION","STRENGTH","CONCEPT_TAGS","SITUATION_TAGS","COVERAGE_TAGS","PRESSURE_TAGS","FILE_NAME"]]
                got.insert(0, "Bucket", bucket)
                call_rows2.append(got)

        if call_rows2:
            call_sheet_matchup = pd.concat(call_rows2, ignore_index=True)

            # CALL (no motions): "<Formation> <Strength>. <PLAY_NAME>"
            base_prefix = (call_sheet_matchup["FORMATION"].fillna("").str.strip() + " " + call_sheet_matchup["STRENGTH"].fillna("").str.strip()).str.strip()
            call_sheet_matchup["CALL"] = base_prefix.where(base_prefix.eq(""), base_prefix + ". ") + call_sheet_matchup["PLAY_NAME"].fillna("")

            st.success("Built matchup-optimized call sheet from library + opponent profile.")
            st.dataframe(call_sheet_matchup)
            st.download_button("⬇️ Download Call_Sheet_Matchup.csv", data=call_sheet_matchup.to_csv(index=False).encode("utf-8"), file_name="Call_Sheet_Matchup.csv", mime="text/csv")

            # Make the rest of the app (one-pager/exports) use this sheet automatically
            call_sheet = call_sheet_matchup
        else:
            st.info("No matchup picks found — add tags like 'vs Man / vs C3 / vs Quarters / vs Blitz' in your play index.")
try:
    if 'call_sheet' in locals() and isinstance(call_sheet, pd.DataFrame) and not call_sheet.empty:
        st.subheader("Printable One-Pager")
        st.caption("Thumbnails match by FORMATION + PLAY NAME (motions ignored). Provide FILE_NAME to override.")
        include_imgs = st.checkbox("Include play art (from uploaded images)", True)
        thumb_px = st.slider("Thumbnail size (px)", 60, 140, 90, 10)
        images_map = st.session_state.PLAYBOOK.get('images', {})

        def _guess_mime(name):
            n = str(name or '').lower()
            if n.endswith(('.jpg','.jpeg')): return 'image/jpeg'
            if n.endswith('.webp'): return 'image/webp'
            return 'image/png'

        def _canon(s: str) -> str:
            import re, unicodedata
            t = unicodedata.normalize('NFKD', str(s or '')).encode('ascii','ignore').decode('ascii')
            t = t.lower().replace('&', 'and')
            return re.sub(r'[^a-z0-9]+', '', t)

        def _strip_motions(txt: str) -> str:
            # remove tokens like "y-over", "x-yo-yo" etc.
            if not txt: return ''
            letters = set("XYHZF")
            known = {"OVER","RETURN","FLY","ORBIT","CLOSE","FLASH","YOYO","YO-YO","TIP"}
            kept = []
            for tok in str(txt).split():
                if '-' in tok:
                    a, b = tok.split('-', 1)
                    if len(a) == 1 and a.upper() in letters and b.replace('-', '').upper() in known:
                        continue  # drop motion
                kept.append(tok)
            return ' '.join(kept).strip()

        def _after_dot(s: str) -> str:
            s = str(s or '')
            return s.split('. ', 1)[1] if '. ' in s else s

        def _find_best_image_for(row) -> str | None:
            # 1) Honor explicit FILE_NAME
            fn = str(row.get('FILE_NAME') or '').strip()
            if fn and fn in images_map:
                return fn

            call = row.get('CALL','') or ''
            pname = row.get('PLAY_NAME','') or ''
            formation = str(row.get('FORMATION','') or '')
            strength = str(row.get('STRENGTH','') or '')

            base_from_call = _strip_motions(_after_dot(call))
            base_from_pname = _after_dot(pname)

            candidate_texts = []
            for base in [base_from_call, base_from_pname]:
                if base:
                    candidate_texts.append(f"{formation} {base}")
                    if strength:
                        candidate_texts.append(f"{formation} {strength} {base}")
                    candidate_texts.append(base)  # fallback
            if pname:
                candidate_texts.append(pname.replace('. ', ' '))

            from difflib import SequenceMatcher
            keynorms = [_canon(t) for t in candidate_texts if t]
            filenorm = {fname: _canon(Path(fname).stem) for fname in images_map.keys()}

            best_name, best_score = None, 0.0
            play_piece = _canon(base_from_call or base_from_pname)
            form_piece = _canon(formation + (' ' + strength if strength else ''))

            for fname, fstem in filenorm.items():
                for kn in keynorms:
                    if not kn: continue
                    score = SequenceMatcher(None, fstem, kn).ratio()
                    if play_piece and play_piece in fstem: score += 0.25
                    if form_piece and form_piece in fstem: score += 0.20
                    if play_piece and form_piece and (play_piece in fstem and form_piece in fstem): score += 0.20
                    if kn in fstem or fstem in kn: score += 0.10
                    if score > best_score:
                        best_score, best_name = score, fname

            return best_name if (best_name and best_score >= 0.55) else None

        def _img_tag_for_row(row):
            if not images_map: return ''
fname = _find_best_image_for(row)
if not fname: return ''
b = images_map.get(fname)
if not b: return ''

            mime = _guess_mime(fname)
            b64 = base64.b64encode(b).decode('utf-8')
            return f'<img class="thumb" src="data:{mime};base64,{b64}" />'

        def _esc(s):
            try:
                import html as _html
                return _html.escape('' if s is None else str(s))
            except Exception:
                return str(s)

        parts = []
        parts.append(f"""
        <style>
        @media print {{
          @page {{ size: Letter landscape; margin: 0.35in; }}
          body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
        }}
        body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
        .bucket {{ margin-bottom: 12px; border: 1px solid #ddd; border-radius: 8px; }}
        .bktitle {{ background:#0f172a; color:white; padding:6px 10px; font-weight:700; }}
        .row {{ display:flex; align-items:center; gap:8px; padding:6px 10px; border-top:1px solid #eee; }}
        .thumb {{ width: {thumb_px}px; height:{thumb_px}px; object-fit:contain; border:1px solid #ccc; border-radius:6px; }}
        .main {{ font-weight:700; font-size:14px; }}
        .meta {{ font-size:12px; color:#334155; }}
        </style>
        """)

        for bucket, dfb in call_sheet.groupby('Bucket', sort=False):
            parts.append(f'<div class="bucket"><div class="bktitle">{_esc(bucket)}</div>')
            for _, r in dfb.iterrows():
                call = _esc(r.get('CALL',''))
                form = _esc(r.get('FORMATION','')); strn = _esc(r.get('STRENGTH',''))
                pers = _esc(r.get('PERSONNEL',''))
                recip = _esc(r.get('RECIPIENT_LETTER',''))
                cov = _esc(r.get('COVERAGE_TAGS','')); press = _esc(r.get('PRESSURE_TAGS',''))
                situ = _esc(r.get('SITUATION_TAGS',''))
                concept = _esc(r.get('CONCEPT_TAGS',''))
                img = _img_tag_for_row(r)

                meta_bits = [x for x in [
                    f"{form} {strn}".strip(),
                    f"Pers {pers}" if pers else '',
                    f"Recipient {recip}" if recip else '',
                    situ
                ] if x]
                meta_line = ' | '.join(meta_bits)
                tags_line = ' • '.join([x for x in [concept, cov, press] if x])

                parts.append(f'''
                  <div class="row">
                    {img if img else ''}
                    <div>
                      <div class="main">{call}</div>
                      <div class="meta">{_esc(meta_line)}</div>
                      {f'<div class="meta">{_esc(tags_line)}</div>' if tags_line else ''}
                    </div>
                  </div>
                ''')
            parts.append('</div>')
        html_str = "\n".join(parts)

        st.components.v1.html(html_str, height=800, scrolling=True)
        st.download_button("⬇️ Download OC_OnePager.html", data=html_str.encode('utf-8'), file_name="OC_OnePager.html", mime="text/html")
except Exception as _e:
    st.warning(f"One-Pager build skipped: {_e}")
