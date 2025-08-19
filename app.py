import io, zipfile, base64, json
from pathlib import Path
from typing import List, Optional
import urllib.request, urllib.error

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------
# Hudl Tendency Analyzer (Universal Language)
# Compact refactor: same features, fewer lines via helpers
# -----------------------

st.set_page_config(page_title="Hudl Tendency Analyzer", layout="wide")
st.title("🏈 Hudl Tendency Analyzer & Game-Plan Builder — Universal Terminology")
st.caption("Upload a Hudl-style CSV (or Excel) to get overall & situational tendencies, diagnostics, and auto-generated game-plan notes using universal football language.")

# -----------------------
# Config & constants
# -----------------------
PRIMARY_COLS = [
    "PLAY_NUM","ODK","DN","DIST","YARD_LN","HASH","OFF_FORM","OFF_STR","BACKFIELD",
    "OFF_PLAY","PLAY_TYPE","PLAY_DIR","RESULT","GN_LS","EFF","DEF_FRONT","DEF_STR",
    "BLITZ","COVERAGE","QTR","MOTION","MOTION_DIR","PASSER","RECEIVER","PLAY_TYPE_RPS",
    "RUSHER","TEAM"
]
ALIAS_MAP = {"YARD LN":"YARD_LN","GN/LS":"GN_LS","PLAY #":"PLAY_NUM","PLAY#":"PLAY_NUM","PLAY NO":"PLAY_NUM","PLAY DIR":"PLAY_DIR"}
DIST_BUCKETS = [(0,3,"short (1-3)"),(4,6,"medium (4-6)"),(7,10,"long (7-10)"),(11,999,"very long (11+)")]

OFF_PRESSURE_ANSWERS = ["RB/WR/TE screens","quick game (slant/flat/hitch)","hot throws vs pressure","max-protect shot (7-man)"]
OFF_MAN_BEATERS = ["Mesh (crossers)","Rub/stack releases","Slant/Flat & Option routes","Back-shoulder fade"]
OFF_C3_BEATERS = ["3-level Flood (Sail)","Curl/Flat","Dagger (seam+dig)","Seam shots vs MOFC"]
OFF_C4_BEATERS = ["Posts & benders","Scissors (Post+Corner)","Flood variations","Deep over (play-action)"]
OFF_SCREEN_FAMILY = ["RB screen","WR bubble/tunnel","TE screen"]
RUN_GAME = ["Inside Zone","Outside Zone / Stretch","Power","Counter","Iso / Lead","Trap","Pin-pull / Toss"]

PLAYBOOK_COLS = ["PLAY_NAME","PERSONNEL","FORMATION","STRENGTH","CONCEPT_TAGS","SITUATION_TAGS","COVERAGE_TAGS","PRESSURE_TAGS","FILE_NAME"]

# -----------------------
# Cached templates
# -----------------------
@st.cache_data(show_spinner=False)
def template_csv_bytes() -> bytes:
    return pd.DataFrame(columns=PRIMARY_COLS).to_csv(index=False).encode("utf-8")

@st.cache_data(show_spinner=False)
def play_index_template_bytes() -> bytes:
    return pd.DataFrame(columns=PLAYBOOK_COLS).to_csv(index=False).encode("utf-8")

# -----------------------
# Helpers — IO & transforms
# -----------------------
def _coerce_gain(val):
    try:
        return float(val)
    except Exception:
        try:
            return float(str(val).replace("+", "").strip())
        except Exception:
            return np.nan


def _to_bucket_dist(dist):
    try:
        d = int(float(dist))
    except Exception:
        return "unknown"
    for lo, hi, name in DIST_BUCKETS:
        if lo <= d <= hi:
            return name
    return "unknown"


def _left_mid_right(s) -> str:
    if pd.isna(s):
        return "Unknown"
    try:
        si = int(float(s));  return {-1:"Left",0:"Middle",1:"Right"}.get(si, "Unknown")
    except Exception:
        pass
    s = str(s).strip().lower()
    if not s: return "Unknown"
    if s.startswith("l"): return "Left"
    if s.startswith("r"): return "Right"
    if s.startswith(("m","i")) or s in {"mid","middle","inside","in","center","ctr"}: return "Middle"
    return "Unknown"


def _normalize_play_type(row: pd.Series) -> str:
    v = str(row.get("PLAY_TYPE_RPS") or "").strip().upper()
    if v in {"R","RUN"}: return "Run"
    if v in {"P","PASS"}: return "Pass"
    if v in {"S","SCREEN"}: return "Screen"
    t = str(row.get("PLAY_TYPE") or "").lower()
    if "screen" in t: return "Screen"
    if "pass" in t: return "Pass"
    if "run" in t or t in {"ko","po","rush"}: return "Run"
    op = str(row.get("OFF_PLAY") or "").lower()
    if any(k in op for k in ["screen","bubble","tunnel"]): return "Screen"
    if any(k in op for k in ["pass","boot","play-action","pa "]): return "Pass"
    if any(k in op for k in ["zone","power","counter","iso","trap","toss","draw","run"]): return "Run"
    return "Unknown"


def _success_row(r: pd.Series):
    dn, dist, g = r.get("DN"), r.get("DIST"), r.get("GN_LS")
    if pd.isna(dn) or pd.isna(dist) or pd.isna(g): return np.nan
    try: dn=int(dn); dist=float(dist); g=float(g)
    except Exception: return np.nan
    return (g >= 0.5*dist) if dn==1 else ((g >= 0.7*dist) if dn==2 else (g >= dist))


def read_table(upload) -> pd.DataFrame:
    if not upload: return pd.DataFrame()
    try:
        return pd.read_excel(upload, engine="openpyxl") if str(upload.name).lower().endswith(".xlsx") else pd.read_csv(upload)
    except Exception:
        try: return pd.read_csv(upload)
        except Exception: return pd.DataFrame()


def prep_hudl(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df is False: return pd.DataFrame()
    df = df.rename(columns={k:v for k,v in ALIAS_MAP.items() if k in df.columns}).copy()
    for c in PRIMARY_COLS:
        if c not in df.columns: df[c] = np.nan
    for c in ["DN","DIST","YARD_LN","QTR","GN_LS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "GN/LS" in df.columns and df["GN_LS"].isna().all():
        df["GN_LS"] = df["GN/LS"].apply(_coerce_gain)

    df["PLAY_TYPE_NORM"] = df.apply(_normalize_play_type, axis=1)
    df["DIR_LMR"] = df["PLAY_DIR"].apply(_left_mid_right)
    df["DIST_BUCKET"] = df["DIST"].apply(_to_bucket_dist)
    df["HASH_N"] = df["HASH"].astype(str).str.upper().str[0].map({"L":"L","R":"R","M":"M"}).fillna("U")
    df["RED_ZONE"] = df["YARD_LN"].apply(lambda y: "yes" if (not pd.isna(y) and 1 <= y <= 20) else "no")
    def _field_zone(y):
        if pd.isna(y): return "unknown"
        if y <= -90: return "own 10 and in"
        if -89 <= y <= -21: return "own territory (21-49)"
        if -20 <= y <= -1: return "near midfield (own 1-20)"
        if 0 <= y <= 20: return "midfield (50 +/- 20)"
        if 21 <= y <= 35: return "plus territory (35-21)"
        if 36 <= y <= 50: return "scoring fringe (20-34)"
        if 1 <= y <= 20: return "red zone (20 and in)"
        return "unknown"
    df["FIELD_ZONE"] = df["YARD_LN"].apply(_field_zone)

    if "SUCCESS" not in df.columns:
        df["SUCCESS"] = df.apply(_success_row, axis=1)
    df["EXPLOSIVE"] = False
    df.loc[(df["PLAY_TYPE_NORM"]=="Run") & (df["GN_LS"] >= 10), "EXPLOSIVE"] = True
    df.loc[(df["PLAY_TYPE_NORM"]=="Pass") & (df["GN_LS"] >= 15), "EXPLOSIVE"] = True
    return df


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
    tbl = g.groupby(dims, dropna=False)["BLITZ_FLAG"].agg(["mean","count"]).reset_index().rename(columns={"mean":"blitz_rate","count":"plays"})
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


def serialize_playbook(pb: dict) -> dict:
    out = {'plays': pb.get('plays', []), 'images': {}, 'sheets_csv_url': st.session_state.get('SHEETS_CSV_URL',''), 'sheets_write_url': st.session_state.get('SHEETS_WRITE_URL',''), 'favorites': sorted(st.session_state.get('FAVORITES', set()))}
    for fname, b in pb.get('images', {}).items():
        try: out['images'][fname] = base64.b64encode(b).decode('utf-8')
        except Exception: pass
    return out


def deserialize_playbook(pb_json: dict) -> dict:
    pb = {'plays': pb_json.get('plays', []), 'images': {}}
    for fname, s in pb_json.get('images', {}).items():
        try: pb['images'][fname] = base64.b64decode(s.encode('utf-8'))
        except Exception: pass
    st.session_state['SHEETS_CSV_URL'] = pb_json.get('sheets_csv_url', '')
    st.session_state['SHEETS_WRITE_URL'] = pb_json.get('sheets_write_url', '')
st.session_state['FAVORITES'] = set(pb_json.get('favorites', []))
return pb


def push_playbook_to_webhook(url: str, rows: list) -> str:
    payload = json.dumps({'rows': rows, 'replace': True, 'columns': PLAYBOOK_COLS}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode('utf-8')

# -----------------------
# UI helpers
# -----------------------
def ui_table(title: str, df: pd.DataFrame, chart: bool=False, pivot=None):
    st.markdown(f"**{title}**")
    if df is None or len(df)==0:
        st.info("No data")
        return
    st.dataframe(df)
    if chart and pivot is not None:
        try:
            st.bar_chart(df.pivot_table(**pivot))
        except Exception:
            pass


def handle_image_uploads(files):
    if not files:
        return
    added = added_from_zip = skipped = dup = 0
    images = st.session_state.PLAYBOOK.setdefault('images', {})
    exts = {'.png', '.jpg', '.jpeg', '.webp'}

    for f in files:
        fname = f.name
        if fname.lower().endswith('.zip'):
            try:
                data = f.read()
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for zi in zf.infolist():
                        if zi.is_dir():
                            continue
                        # skip macOS resource files and folders
                        if zi.filename.startswith('__MACOSX/') or Path(zi.filename).name.startswith('._'):
                            continue
                        ext = Path(zi.filename).suffix.lower()
                        if ext not in exts:
                            skipped += 1
                            continue
                        try:
                            buf = zf.read(zi)
                        except Exception:
                            skipped += 1
                            continue
                        base = Path(zi.filename).name
                        stem = Path(base).stem
                        key = base
                        idx = 1
                        # de-duplicate filenames so multiple images with the same name don't overwrite
                        while key in images:
                            dup += 1
                            key = f"{stem}_{idx}{ext}"
                            idx += 1
                        images[key] = buf
                        added += 1
                        added_from_zip += 1
            except Exception as e:
                st.error(f"Couldn't read zip '{fname}': {e}")
        else:
            try:
                ext = Path(fname).suffix.lower()
                if ext in exts:
                    base = Path(fname).name
                    stem = Path(base).stem
                    key = base
                    idx = 1
                    while key in images:
                        dup += 1
                        key = f"{stem}_{idx}{ext}"
                        idx += 1
                    images[key] = f.read()
                    added += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    msg = f"Stored {added} image(s)" + (f" ({added_from_zip} from zip)" if added_from_zip else '')
    if dup:
        msg += f"; handled {dup} duplicate name(s)"
    if skipped:
        msg += f"; skipped {skipped} non-image or invalid file(s)"
    st.success(msg)

# -----------------------
# Suggestions builder (unchanged logic)
# -----------------------
def safe_rate(n, d):
    return (n / d) if d else 0.0


def build_suggestions(overall, by_down, by_dist, by_form, blitz_3rd, cov_3rd, hash_tbl, motion_tbl, sample_size, sr_overall, xpl_overall):
    """Robust to empty/missing columns in grouped tables."""
    suggestions = []

    # Sample size & overall mix
    if sample_size < 25:
        suggestions.append(f"Warning: small sample ({sample_size} plays). Treat tendencies with caution.")
    total = overall["plays"].sum() if len(overall) else 0
    rp = overall[overall.get("PLAY_TYPE_NORM", pd.Series(index=overall.index)).isin(["Run","Pass","Screen"]) ] if len(overall) else overall
    run_p = (rp[rp.get("PLAY_TYPE_NORM")=="Run"]["plays"].sum() / total) if total else 0
    pass_p = (rp[rp.get("PLAY_TYPE_NORM")=="Pass"]["plays"].sum() / total) if total else 0
    screen_p = (rp[rp.get("PLAY_TYPE_NORM")=="Screen"]["plays"].sum() / total) if total else 0
    if run_p >= 0.60:
        suggestions.append(f"Run-heavy profile overall ({run_p:.0%}). Def: extra fitter on early downs; Off: play-action shots.")
    elif pass_p >= 0.60:
        suggestions.append(f"Pass-heavy profile overall ({pass_p:.0%}). Def: simulated pressures, late rotation from two-high.")
    if screen_p >= 0.10:
        suggestions.append(f"Screens appear {screen_p:.0%}. DL retrace; keep RB/WR/TE screen menu ready.")

    # Efficiency
    if not np.isnan(sr_overall):
        suggestions.append(f"Estimated success rate: {sr_overall:.0%} (success = gain ≥ yard-to-go on 1st/2nd, or conversion on 3rd).")
    if not np.isnan(xpl_overall):
        suggestions.append(f"Explosive rate: {xpl_overall:.0%} (10+ rush / 15+ pass).")

    # By Down
    if isinstance(by_down, pd.DataFrame) and len(by_down) and "DN" in by_down.columns:
        for d in [1,2,3]:
            sub = by_down[by_down["DN"]==d]
            if len(sub):
                types = set(sub.get("PLAY_TYPE_NORM", pd.Series(dtype=object)))
                r = sub[sub.get("PLAY_TYPE_NORM")=="Run"]["%"].sum() if "Run" in types else 0
                p = sub[sub.get("PLAY_TYPE_NORM")=="Pass"]["%"].sum() if "Pass" in types else 0
                if d==1 and r>=60:
                    suggestions.append("1st down leans run. Fit downhill; be alert for play-action off base runs.")
                if d==3 and p>=70:
                    suggestions.append("3rd down leans pass. Def: simulated pressure, play the sticks. Off: quick man/zone beaters.")

    # 3rd & Distance
    if isinstance(by_dist, pd.DataFrame) and len(by_dist) and all(c in by_dist.columns for c in ["DN","DIST_BUCKET","PLAY_TYPE_NORM","%"]):
        td = by_dist[by_dist["DN"]==3]
        if len(td):
            lp = td[(td["DIST_BUCKET"]=="long (7-10)") & (td["PLAY_TYPE_NORM"]=="Pass")]
            if len(lp) and lp["%"].iloc[0] >= 70:
                suggestions.append("3rd & 7–10 high pass tendency. Off: Flood/Sail or Dagger. Def: mug A-gaps, spin to 3-robber.")
            vl = td[(td["DIST_BUCKET"]=="very long (11+)") & (td["PLAY_TYPE_NORM"]=="Pass")]
            if len(vl) and vl["%"].iloc[0] >= 80:
                suggestions.append("3rd & 11+ = must-pass. Spy QB if mobile; Off: max-protect shots vs soft zone.")

    # Top formation
    if isinstance(by_form, pd.DataFrame) and len(by_form) and all(c in by_form.columns for c in ["OFF_FORM","PLAY_TYPE_NORM","%","plays"]):
        top_form = by_form.groupby(["OFF_FORM"])['plays'].sum().reset_index().sort_values("plays", ascending=False).head(1)
        if len(top_form):
            f = str(top_form["OFF_FORM"].iloc[0])
            f_tbl = by_form[by_form["OFF_FORM"]==f]
            types = set(f_tbl.get("PLAY_TYPE_NORM", pd.Series(dtype=object)))
            run_bias = f_tbl[f_tbl.get("PLAY_TYPE_NORM")=="Run"]["%"].sum() if "Run" in types else 0
            if run_bias >= 65:
                suggestions.append(f"In {f}, strong run tendency (~{run_bias:.0f}%). Def: set edges, close interior gaps. Off: play-action counters from same look.")

    # 3rd down pressure & coverage
    if isinstance(blitz_3rd, pd.DataFrame) and len(blitz_3rd) and "blitz_rate" in blitz_3rd.columns:
        if (blitz_3rd["blitz_rate"] >= 0.35).any():
            suggestions.append("They pressure on 3rd (≥35%). Off: screens, quick game, hot throws; consider max-protect shots.")
    if isinstance(cov_3rd, pd.DataFrame) and len(cov_3rd) and all(c in cov_3rd.columns for c in ["COVERAGE_N","plays"]):
        top_cov = cov_3rd.groupby(["COVERAGE_N"])['plays'].sum().reset_index().sort_values("plays", ascending=False).head(1)
        if len(top_cov):
            cov = top_cov["COVERAGE_N"].iloc[0]
            if cov in {"COVER 1","C1","MAN","COVER1"}: suggestions.append("3rd = Man. Off: Mesh, rub/stack, option routes, back-shoulder.")
            if cov in {"COVER 3","C3","THREE","COVER3"}: suggestions.append("3rd = Cover 3 (MOFC). Off: Flood/Sail, Curl-Flat, Dagger; attack seams/curl window.")
            if cov in {"COVER 4","C4","QUARTERS"}: suggestions.append("3rd = Quarters. Off: posts & benders, scissors, deep overs.")

    # Hash tendencies
    if isinstance(hash_tbl, pd.DataFrame) and len(hash_tbl) and all(c in hash_tbl.columns for c in ["HASH_N","%"]):
        left = hash_tbl[hash_tbl["HASH_N"]=="L"]["%"].sum() if (hash_tbl["HASH_N"]=="L").any() else 0
        right = hash_tbl[hash_tbl["HASH_N"]=="R"]["%"].sum() if (hash_tbl["HASH_N"]=="R").any() else 0
        if left >= 55: suggestions.append("Left-hash bias. Off: set strength to field; Def: set strength to boundary, fit fast to field.")
        if right >= 55: suggestions.ap

# -----------------------
# Session defaults
# -----------------------
if 'PLAYBOOK' not in st.session_state:
    st.session_state.PLAYBOOK = {'plays': [], 'images': {}}
if 'SHEETS_CSV_URL' not in st.session_state:
    st.session_state.SHEETS_CSV_URL = ''
if 'SHEETS_WRITE_URL' not in st.session_state:
    st.session_state.SHEETS_WRITE_URL = ''
if 'CONCEPT_FORMATION_RULES' not in st.session_state:
    st.session_state.CONCEPT_FORMATION_RULES = {'TRAIN': ['TRIPS']}
if 'SCREEN_RECIPIENT_ORDER' not in st.session_state:
    st.session_state.SCREEN_RECIPIENT_ORDER = ['Y','Z','H','X','F']
if 'FAVORITES' not in st.session_state:
    st.session_state.FAVORITES = set()

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Controls")
st.sidebar.text_input("Team label (optional)", value="Opponent")
show_charts = st.sidebar.checkbox("Show charts", value=True)
st.sidebar.download_button("Download CSV Template", data=template_csv_bytes(), file_name="hudl_template.csv", mime="text/csv")

# -----------------------
# Playbook Library & Google Sheets
# -----------------------
st.subheader("Playbook Library — Upload once, reuse forever")
col_pb1, col_pb2 = st.columns([2,1])
with col_pb1:
    st.markdown("**A) Load or build your library**")
    pb_json = st.file_uploader("Load playbook.json (optional)", type=["json"], key="pbjson")
    if pb_json:
        try:
            st.session_state.PLAYBOOK = deserialize_playbook(json.loads(pb_json.read().decode('utf-8')))
            st.success(f"Loaded {len(st.session_state.PLAYBOOK.get('plays', []))} plays from your library.")
        except Exception as e:
            st.error(f"Couldn't load playbook.json: {e}")

    st.markdown("**Upload play images** (PNG/JPG/WEBP or ZIP). Name files to match FILE_NAME in your index, or exactly the PLAY_NAME.")
    uploads = st.file_uploader("Add play screenshots/diagrams (PNG/JPG/WEBP or .zip)", type=["png","jpg","jpeg","webp","zip"], accept_multiple_files=True, key="pbimgs")
    if uploads:
        handle_image_uploads(uploads)
    # Quick sanity check so you can confirm images were stored
if st.session_state.PLAYBOOK.get('images'):
    st.caption(f"Library images: {len(st.session_state.PLAYBOOK['images'])}")

# ⭐ Favorites UI (persisted in playbook.json)
with st.expander("⭐ Favorites (persisted)", expanded=False):
    lib_names = [p.get('PLAY_NAME','') for p in st.session_state.PLAYBOOK.get('plays',[])]
    current = sorted(st.session_state.get('FAVORITES', set()))
    picked = st.multiselect("Mark favorite plays", options=lib_names, default=current)
    st.session_state.FAVORITES = set(picked)

st.download_button("Download Play Index CSV Template", data=play_index_template_bytes(), file_name="play_index_template.csv", mime="text/csv")
    play_index_csv = st.file_uploader("Upload play_index.csv (" + ", ".join(PLAYBOOK_COLS) + ")", type=["csv"], key="pbidx")
    if play_index_csv:
        try:
            df_idx = pd.read_csv(play_index_csv)
            miss = [c for c in PLAYBOOK_COLS if c not in df_idx.columns]
            if miss: st.error("Missing columns: " + ", ".join(miss))
            else:
                st.session_state.PLAYBOOK['plays'] = df_idx.to_dict(orient='records')
                st.success(f"Indexed {len(df_idx)} plays.")
        except Exception as e:
            st.error(f"Could not read play_index.csv: {e}")

    st.markdown("**Save your playbook** (download a .json to reuse later)")
    st.download_button("⬇️ Save Playbook Library (playbook.json)", data=json.dumps(serialize_playbook(st.session_state.PLAYBOOK)).encode('utf-8'), file_name="playbook.json", mime="application/json")

with col_pb2:
    st.markdown("**B) Google Sheets Sync (optional)**")
    st.caption("No installs. Use a Google Sheet as your play index. 'Publish to web' → paste CSV link below.")
    sheets_url = st.text_input("Google Sheets — Published CSV URL (read-only)", value=st.session_state.get('SHEETS_CSV_URL',''), placeholder="https://docs.google.com/...output=csv")
    if st.button("Load from Google Sheets"):
        try:
            df_idx = pd.read_csv(sheets_url)
            miss = [c for c in PLAYBOOK_COLS if c not in df_idx.columns]
            if miss: st.error("Your sheet is missing columns: " + ", ".join(miss))
            else:
                st.session_state.PLAYBOOK['plays'] = df_idx.to_dict(orient='records')
                st.session_state.SHEETS_CSV_URL = sheets_url
                st.success(f"Loaded {len(df_idx)} plays from Google Sheets.")
        except Exception as e:
            st.error(f"Couldn't load Google Sheets CSV: {e}")
    writer_url = st.text_input("Google Apps Script Web App URL (optional)", value=st.session_state.get('SHEETS_WRITE_URL',''), placeholder="https://script.google.com/.../exec")
    if st.button("Push library to Google Sheets"):
        try:
            resp = push_playbook_to_webhook(writer_url, st.session_state.PLAYBOOK.get('plays', []))
            st.session_state.SHEETS_WRITE_URL = writer_url
            st.success("Pushed to Google Sheets. Response: " + str(resp)[:200])
        except Exception as e:
            st.error(f"Push failed: {e}")

    st.markdown("**Quick Add (single play)**")
    with st.form("quick_add_play"):
        vals = {
            'PLAY_NAME': st.text_input("Play Name"),
            'PERSONNEL': st.text_input("Personnel (e.g., 11, 12, 20)"),
            'FORMATION': st.text_input("Formation (e.g., Trips, Bunch, Singleback)"),
            'STRENGTH': st.text_input("Strength (Right/Left/Boundary/Field)"),
            'CONCEPT_TAGS': st.text_input("Concept tags (comma-separated: Flood, Mesh, Curl-Flat, PA, Boot, RPO, IZ, OZ, Power, Counter)"),
            'SITUATION_TAGS': st.text_input("Situation tags (3rd&7-10, 1st&10, RZ, 2-minute, vs Nickel)"),
            'COVERAGE_TAGS': st.text_input("Coverage tags (vs Man, vs C3, vs Quarters)"),
            'PRESSURE_TAGS': st.text_input("Pressure tags (vs Blitz, vs 5-man, vs DB blitz)"),
            'FILE_NAME': st.text_input("Image file name (must match an uploaded image)")
        }
        if st.form_submit_button("Add Play"):
            st.session_state.PLAYBOOK['plays'].append(vals)
            st.success(f"Added play: {vals['PLAY_NAME']}")

# -----------------------
# Call Sheet Rules
# -----------------------
with st.expander("Call Sheet Rules (formations & screens)"):
    lib_for_opts = pd.DataFrame(st.session_state.PLAYBOOK.get('plays', []))
    form_opts = sorted(set(str(x).upper() for x in lib_for_opts.get('FORMATION', pd.Series(dtype=str)).dropna().unique())) or [
        "DICE","DECK","DOUBLES","DOS","TRIPS","TWIG","TREY","TRIO","BUNCH","BUNDLE","GANG","GLOCK","GAUGE","EMPTY","EGO"
    ]
    rules = st.session_state.get('CONCEPT_FORMATION_RULES', {})
    for concept_key, label in [('TRAIN','TRAIN (tunnel screen)'),('VIPER','VIPER (swing screen)'),('UNICORN','UNICORN (throwback)'),('UTAH','UTAH (shovel/drag)')]:
        cur = [s.upper() for s in rules.get(concept_key, [])]
        rules[concept_key] = st.multiselect(f"Allowed formations for {label}", options=form_opts, default=cur, key=f"rule_{concept_key}")
    st.session_state.CONCEPT_FORMATION_RULES = rules

    st.markdown("**Screen recipient letters (order of preference)**")
    order_str = st.text_input("Letters in order (comma-separated)", value=",".join(st.session_state.get('SCREEN_RECIPIENT_ORDER', ['Y','Z','H','X','F'])))
    try:
        parsed = [s.strip().upper() for s in order_str.split(',') if s.strip()]
        if parsed: st.session_state.SCREEN_RECIPIENT_ORDER = parsed
    except Exception:
        pass

# -----------------------
# Upload opponent data
# -----------------------
file = st.file_uploader("Upload Hudl-style CSV or Excel (optional)", type=["csv","xlsx"]) 
raw = prep_hudl(read_table(file)) if file else pd.DataFrame(columns=PRIMARY_COLS)
if raw.empty and not file:
    st.info("No opponent data uploaded — that's okay! We'll still build a game plan from your Playbook Library below.")

# -----------------------
# Diagnostics & grouped tables
# -----------------------
st.subheader("Data Diagnostics")
key_cols = ["DN","DIST","YARD_LN","HASH","PLAY_TYPE","PLAY_TYPE_RPS","OFF_FORM","OFF_STR","BACKFIELD","PLAY_DIR","BLITZ","COVERAGE","MOTION","GN_LS"]
if not raw.empty:
    missing = [c for c in key_cols if raw[c].isna().all()]
    st.write("**Non-null counts (how much data do we have by column):**")
    st.dataframe(raw[key_cols].notna().sum().to_frame("non-null").T)
    if missing: st.warning("No data found in: " + ", ".join(missing))

overall = raw.groupby(["PLAY_TYPE_NORM"]).size().reset_index(name="plays") if not raw.empty else pd.DataFrame(columns=["PLAY_TYPE_NORM","plays"]) 
if len(overall): overall["%"] = (100 * overall["plays"] / overall["plays"].sum()).round(1)

sample_size = int(overall["plays"].sum()) if len(overall) else int(len(raw))
sr_overall = raw["SUCCESS"].mean() if (not raw.empty and raw["SUCCESS"].notna().any()) else np.nan
xpl_overall = raw["EXPLOSIVE"].mean() if (not raw.empty and raw["GN_LS"].notna().any()) else np.nan

by_down = tendency_table(raw, ["DN"]) if not raw.empty else pd.DataFrame()
by_dist = tendency_table(raw, ["DN","DIST_BUCKET"]) if not raw.empty else pd.DataFrame()
by_hash = tendency_table(raw, ["HASH_N"]) if not raw.empty else pd.DataFrame()
by_form = tendency_table(raw, ["OFF_FORM","OFF_STR","BACKFIELD"]) if not raw.empty else pd.DataFrame()
by_dir  = tendency_table(raw, ["DIR_LMR"]) if not raw.empty else pd.DataFrame()
by_fz   = tendency_table(raw, ["FIELD_ZONE"]) if not raw.empty else pd.DataFrame()
by_rz   = tendency_table(raw[raw["RED_ZONE"]=="yes"], ["DN"]) if (not raw.empty and (raw["RED_ZONE"]=="yes").any()) else pd.DataFrame()

blitz_3rd = compute_blitz_rate(raw[raw["DN"]==3], ["DIST_BUCKET"]) if (not raw.empty and "DN" in raw.columns) else pd.DataFrame()
cov_3rd   = compute_coverage(raw[raw["DN"]==3], ["DIST_BUCKET"])   if (not raw.empty and "DN" in raw.columns) else pd.DataFrame()

# -----------------------
# Overview
# -----------------------
st.subheader("Overview")
col1, col2, col3 = st.columns(3)
with col1:
    ui_table("Run/Pass/Screen Mix", overall, chart=show_charts, pivot=dict(index="PLAY_TYPE_NORM", values="plays", aggfunc="sum"))
with col2:
    ui_table("Direction (L/M/R)", by_dir)
with col3:
    st.markdown("**Efficiency**")
    eff = []
    if not np.isnan(sr_overall): eff.append(f"Success rate: {sr_overall:.0%}")
    if not np.isnan(xpl_overall): eff.append(f"Explosive rate: {xpl_overall:.0%}")
    if eff: st.write("\\n".join(eff))
    else: st.info("Add GN_LS (gain/loss) to compute success & explosive rates.")

st.divider()

# -----------------------
# Situational Tendencies
# -----------------------
st.subheader("Situational Tendencies")
with st.expander("By Down", expanded=True):
    if len(by_down):
        ui_table("By Down", by_down, chart=show_charts, pivot=dict(index="DN", columns="PLAY_TYPE_NORM", values="%", fill_value=0))
    else:
        st.info("Missing DN or PLAY_TYPE.")
with st.expander("By Down & Distance"):
    ui_table("By Down & Distance", by_dist)
with st.expander("By Formation / Strength / Backfield"):
    ui_table("By Formation / Strength / Backfield", by_form)

c1, c2 = st.columns(2)
with c1:
    ui_table("By Hash", by_hash)
with c2:
    ui_table("By Field Zone", by_fz)

st.markdown("**Red Zone by Down**")
if len(by_rz): st.dataframe(by_rz)
else: st.info("No red-zone snaps detected (YARD_LN 1–20).")

st.subheader("3rd Down Study")
cc1, cc2 = st.columns(2)
with cc1:
    ui_table("Blitz Rate (3rd Down)", blitz_3rd)
with cc2:
    ui_table("Coverage Usage (3rd Down)", cov_3rd)

st.subheader("Motion Usage")
if len(raw):
    mt = raw.copy()
    mt["MOTION_N"] = np.where(mt["MOTION"].astype(str).str.strip().eq(""), "None", "Motion")
    mt = mt.groupby(["MOTION_N","PLAY_TYPE_NORM"], dropna=False).size().reset_index(name="plays")
    den = mt.groupby(["MOTION_N"])['plays'].transform('sum').replace(0, np.nan)
    mt["%"] = ((100 * mt["plays"] / den).round(1)).fillna(0)
    st.dataframe(mt)
else:
    st.info("No motion data found.")

# -----------------------
# Suggestions & quick buckets
# -----------------------
suggestions = build_suggestions(overall, by_down, by_dist, by_form, blitz_3rd, cov_3rd, by_hash, mt if len(raw) else pd.DataFrame(), sample_size, sr_overall, xpl_overall)
st.subheader("Auto-Generated Game-Plan Suggestions (Universal)")
for s in suggestions: st.markdown(f"- {s}")

st.markdown("### Quick Call-Sheet Buckets (Offense — Universal)")
st.markdown("**Vs Pressure (esp. 3rd):** " + ", ".join(OFF_PRESSURE_ANSWERS))
st.markdown("**Vs Man (3rd):** " + ", ".join(OFF_MAN_BEATERS))
st.markdown("**Vs Cover 3 (MOFC):** " + ", ".join(OFF_C3_BEATERS))
st.markdown("**Vs Quarters:** " + ", ".join(OFF_C4_BEATERS))
st.markdown("**Screen Menu:** " + ", ".join(OFF_SCREEN_FAMILY))
st.markdown("**Run Game:** " + ", ".join(RUN_GAME))

# -----------------------
# Playbook-Only Planner
# -----------------------
# -----------------------
# Priority & badge helpers
# -----------------------

def _icons_for_row(r: pd.Series) -> str:
    txt = (" ".join([str(r.get(k,'')) for k in ['CONCEPT_TAGS','SITUATION_TAGS','COVERAGE_TAGS','PRESSURE_TAGS']])).lower()
    icons = []
    if any(k in txt for k in ['quick','stick','snag','rpo','hitch','slant']): icons.append('⚡')
    if any(k in txt for k in ['rz','red zone','money','atm']): icons.append('🔴')
    if any(k in txt for k in ['vs c3','vs cover 3',' cover 3',' c3 ']): icons.append('3️⃣')
    if any(k in txt for k in ['vs blitz',' pressure','blitz']): icons.append('🚨')
    return " ".join(icons)

def _priority_for_row(r: pd.Series, score: Optional[float] = None) -> int:
    base = 1.0
    if r.get('PLAY_NAME') in st.session_state.get('FAVORITES', set()):
        base += 1.5
    ct = str(r.get('CONCEPT_TAGS','')).lower()
    if any(k in ct for k in ['screen','bubble','tunnel']): base += 0.5
    if any(k in ct for k in ['inside zone','iz','outside zone','oz','power','counter']): base += 0.5
    if score is not None:
        try: base += min(2.0, max(0.0, float(score))) * 0.5
        except Exception: pass
    return int(max(1, min(5, round(base))))

st.subheader("Playbook-Only Planner (No Opponent Data)")
lib = pd.DataFrame(st.session_state.PLAYBOOK.get('plays', []))
if lib.empty:
    st.info("Add plays to your Playbook Library above to generate a call sheet without opponent data.")
else:
    def _text_contains(text: str, keywords: List[str]) -> bool:
        hay = str(text or '').lower();  return any(k in hay for k in keywords)

    def _detect_concept(row: pd.Series) -> Optional[str]:
        txt = (str(row.get('CONCEPT_TAGS','')) + ' ' + str(row.get('PLAY_NAME',''))).upper()
        for key in ['TRAIN','VIPER','UNICORN','UTAH']:
            if key in txt: return key
        return None

    def _pick_varied(df: pd.DataFrame, keywords: List[str], limit: Optional[int] = None) -> pd.DataFrame:
        mask = df.apply(lambda r: _text_contains((r.get('CONCEPT_TAGS','') + ' ' + r.get('SITUATION_TAGS','')), keywords), axis=1)
        cand = df[mask].copy() if mask.any() else df.copy()
        picked, used_names, used_forms = [], set(), set()
        for _, r in cand.iterrows():
            name, form = r.get('PLAY_NAME'), r.get('FORMATION')
            if name in used_names or form in used_forms: continue
            picked.append(r); used_names.add(name); used_forms.add(form)
            if limit and len(picked) >= limit: break
        return pd.DataFrame(picked)

    rules = st.session_state.get('CONCEPT_FORMATION_RULES', {})
    lib2 = lib.copy()
    lib2['__CONCEPT__'] = lib2.apply(_detect_concept, axis=1)
    keep = []
    for _, rr in lib2.iterrows():
        ck = rr['__CONCEPT__']
        if ck and rules.get(ck): keep.append(str(rr.get('FORMATION','')).upper() in {s.upper() for s in rules[ck]})
        else: keep.append(True)
    lib2 = lib2[pd.Series(keep, index=lib2.index)].drop(columns=['__CONCEPT__'])

    buckets = {
        "1st & 10": ["1st&10","quick","inside zone","iz","outside zone","oz","power","counter","rpo","play-action","boot"],
        "2nd & medium (4-6)": ["2nd&medium","quick","screen","draw","counter","boot","pa"],
        "3rd & short (1-3)": ["3rd&1-3","hitch","slant","snag","mesh","iso","power","qb sneak"],
        "3rd & medium (4-6)": ["3rd&4-6","snag","smash","flood","curl-flat","mesh","screen"],
        "3rd & long (7-10)": ["3rd&7-10","dagger","flood","smash","screen","wheel"],
        "Red Zone": ["rz","money","atm","smash","snag","fade","rub","pick"],
        "2-minute": ["2-minute","hurry","sideline","dagger","curl-flat","out","arrow"],
    }

    call_rows = []
    for bucket, keys in buckets.items():
        picked = _pick_varied(lib2, [k.lower() for k in keys], limit=None)
        if not picked.empty:
            picked = picked[["PLAY_NAME","PERSONNEL","FORMATION","STRENGTH","CONCEPT_TAGS","SITUATION_TAGS","COVERAGE_TAGS","PRESSURE_TAGS","FILE_NAME"]]
            picked.insert(0, "Bucket", bucket)
            call_rows.append(picked)

    if call_rows:
        call_sheet = pd.concat(call_rows, ignore_index=True)
        # Build CALL & screen recipient logic
        order = st.session_state.get('SCREEN_RECIPIENT_ORDER', ['Y','Z','H','X','F'])
        def _screen_concept(row: pd.Series) -> Optional[str]:
            txt = (str(row.get('CONCEPT_TAGS','')) + ' ' + str(row.get('PLAY_NAME',''))).upper()
            for key in ['TRAIN','VIPER','UNICORN','UTAH']:
                if key in txt: return key
            return 'SCREEN' if 'SCREEN' in txt else None
        call_sheet = call_sheet.copy(); call_sheet["RECIPIENT_LETTER"] = ""
        base = (call_sheet["FORMATION"].fillna("").str.strip() + " " + call_sheet["STRENGTH"].fillna("").str.strip()).str.strip()
        call_sheet["CALL"] = base.where(base.eq(""), base + ". ") + call_sheet["PLAY_NAME"].fillna("")
        sidx = [i for i in call_sheet.index if _screen_concept(call_sheet.loc[i])]
        for j, idx in enumerate(sidx):
            recip = order[j % len(order)]; concept = _screen_concept(call_sheet.loc[idx]) or 'SCREEN'
            form = str(call_sheet.at[idx,'FORMATION'] or '').strip(); strength = str(call_sheet.at[idx,'STRENGTH'] or '').strip()
            prefix = (form + (' ' + strength if strength else '')).strip()
            call_sheet.at[idx,'RECIPIENT_LETTER'] = recip
            call_sheet.at[idx,'CALL'] = (f"{prefix}. " if prefix else '') + f"{recip.lower()}-{concept.lower()}"
        # annotate with icons and priority
        call_sheet['ICONS'] = call_sheet.apply(_icons_for_row, axis=1)
        call_sheet['PRIORITY'] = call_sheet.apply(_priority_for_row, axis=1)
        cols = ["Bucket","PLAY_NAME","PRIORITY","ICONS","PERSONNEL","FORMATION","STRENGTH","CONCEPT_TAGS","SITUATION_TAGS","COVERAGE_TAGS","PRESSURE_TAGS","FILE_NAME","CALL","RECIPIENT_LETTER"]
        call_sheet = call_sheet[[c for c in cols if c in call_sheet.columns]]
        st.dataframe(call_sheet)
        st.download_button("⬇️ Download Call_Sheet_PlaybookOnly.csv", data=call_sheet.to_csv(index=False).encode("utf-8"), file_name="Call_Sheet_PlaybookOnly.csv", mime="text/csv")

        # ----- Printable One-Pager (Ratings + Badges, image on right) -----
        try:
            images_map = st.session_state.PLAYBOOK.get('images', {})
            include_imgs = st.checkbox("Include play art (from uploaded images)", True, key="onepager_imgs")
            thumb_px = st.slider("Thumbnail size (px)", 60, 140, 90, 10, key="onepager_thumb")
            def _canon(s: str) -> str:
                import re, unicodedata
                t = unicodedata.normalize('NFKD', str(s or '')).encode('ascii','ignore').decode('ascii')
                return re.sub(r'[^a-z0-9]+','', t.lower())
            def _guess_mime(name):
                n = str(name or '').lower()
                if n.endswith(('.jpg','.jpeg')): return 'image/jpeg'
                if n.endswith('.webp'): return 'image/webp'
                return 'image/png'
            def _find_image(row) -> Optional[str]:
                fn = str(row.get('FILE_NAME') or '').strip()
                if fn and fn in images_map: return fn
                # simple fuzzy: match by play name against filename stem
                pname = _canon(row.get('PLAY_NAME',''))
                best,score=None,0.0
                for fname in images_map.keys():
                    stem = _canon(Path(fname).stem)
                    sc = 1.0 if pname and pname in stem else 0.0
                    if sc>score: best,score=fname,sc
                return best if score>=1.0 else None
            def _img_tag(row):
                if not include_imgs or not images_map: return ''
                fn = _find_image(row)
                if not fn: return ''
                b = images_map.get(fn)
                if not b: return ''
                import base64
                b64 = base64.b64encode(b).decode('utf-8')
                return f"<img class='thumb' src='data:{_guess_mime(fn)};base64,{b64}' />"
            def _esc(x):
                import html as _html
                return _html.escape('' if x is None else str(x))
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
            .row {{ display:flex; align-items:flex-start; gap:10px; padding:8px 10px; border-top:1px solid #eee; }}
            .info {{ flex:1 1 auto; min-width:0; }}
            .thumb {{ width: {thumb_px}px; height:{thumb_px}px; object-fit:contain; border:1px solid #ccc; border-radius:6px; margin-left:auto; }}
            .main {{ font-weight:700; font-size:14px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
            .stars {{ color:#f59e0b; }}
            .badges {{ opacity:0.9; }}
            .meta {{ font-size:12px; color:#334155; }}
            </style>
            """)
            for bucket, dfb in call_sheet.groupby('Bucket', sort=False):
                parts.append(f"<div class='bucket'><div class='bktitle'>{_esc(bucket)}</div>")
                for _, r in dfb.iterrows():
                    name=_esc(r.get('PLAY_NAME',''))
                    stars='⭐'*int(r.get('PRIORITY',1))
                    badges=_esc(r.get('ICONS',''))
                    form=_esc(r.get('FORMATION','')); strn=_esc(r.get('STRENGTH',''))
                    pers=_esc(r.get('PERSONNEL',''))
                    cov=_esc(r.get('COVERAGE_TAGS','')); press=_esc(r.get('PRESSURE_TAGS',''))
                    situ=_esc(r.get('SITUATION_TAGS',''))
                    concept=_esc(r.get('CONCEPT_TAGS',''))
                    meta_bits=[x for x in [f"{form} {strn}".strip(), f"Pers {pers}" if pers else '', situ] if x]
                    meta_line=' | '.join(meta_bits)
                    tags_line=' • '.join([x for x in [concept, cov, press] if x])
                    img=_img_tag(r)
                    parts.append(f"""
                      <div class='row'>
                        <div class='info'>
                          <div class='main'>{name} <span class='stars'>{stars}</span> <span class='badges'>{badges}</span></div>
                          <div class='meta'>{_esc(meta_line)}</div>
                          {f"<div class='meta'>{_esc(tags_line)}</div>" if tags_line else ''}
                        </div>
                        {img}
                      </div>
                    """)
                parts.append("</div>")
            html_str = "
".join(parts)
            st.components.v1.html(html_str, height=800, scrolling=True)
            st.download_button("⬇️ Download OC_OnePager_v2.html", data=html_str.encode('utf-8'), file_name="OC_OnePager_v2.html", mime="text/html")
        except Exception as _e:
            st.warning(f"One-Pager build skipped: {_e}")
    else:
        st.info("Could not find plays matching the standard buckets. Add situation tags like '1st&10', '3rd&7-10', or concept tags (Snag, Flood, Mesh, Smash, Dagger, Screen, IZ/OZ/Power/Counter).")

# -----------------------
# Offense-Focused Matchup Builder
# -----------------------
with st.expander("Offense-Focused Matchup Builder (Our O vs Their D)", expanded=False):
    st.caption("Upload defensive data for the opponent and (optionally) our offensive data. We'll rank your Playbook Library plays by matchup fit: our best concepts vs their most-used coverages & pressure.")
    opp_file = st.file_uploader("Opponent DEF Hudl CSV/Excel", type=["csv","xlsx"], key="oppdef")
    off_file = st.file_uploader("Our OFF Hudl CSV/Excel (optional)", type=["csv","xlsx"], key="ouroff")
    opp_df = prep_hudl(read_table(opp_file))
    off_df = prep_hudl(read_table(off_file))

    if not lib.empty and not opp_df.empty:
        def _cov_norm(s: str) -> str:
            s = str(s or '').upper()
            if any(k in s for k in ["COVER 1","C1","MAN"]): return "MAN"
            if any(k in s for k in ["COVER 3","C3","THREE"]): return "C3"
            if any(k in s for k in ["COVER 4","C4","QUARTERS"]): return "C4"
            if any(k in s for k in ["COVER 2","C2","TWO"]): return "C2"
            return "UNK"
        oc = opp_df.copy(); oc["COVN"] = oc.get("COVERAGE", pd.Series(index=oc.index)).astype(str).apply(_cov_norm)
        cov_dist = oc["COVN"].value_counts(normalize=True).to_dict()
        b3_tbl = compute_blitz_rate(opp_df[opp_df["DN"]==3], ["DIST_BUCKET"]) if ("DIST_BUCKET" in opp_df.columns and "DN" in opp_df.columns) else pd.DataFrame()
        opp_blitz3 = float(b3_tbl["blitz_rate"].mean()) if len(b3_tbl) else 0.0
        concept_map = {"mesh":["mesh"],"smash":["smash"],"flood":["flood","sail"],"dagger":["dagger"],"curlflat":["curl flat","curl-flat","curl","flat"],"screen":["screen","bubble","tunnel"],"iz":["inside zone","iz"],"oz":["outside zone","oz","stretch"],"power":["power"],"counter":["counter"],"wheel":["wheel"],"hitch":["hitch"],"arrow":["arrow"],"stick":["stick","snag"]}
        def _infer_concepts(txt: str) -> List[str]:
            t = str(txt or '').lower();  hits = [k for k,kws in concept_map.items() if any(kw in t for kw in kws)];  return hits or ["other"]
        if not off_df.empty:
            txtcol = off_df.get("OFF_PLAY") if "OFF_PLAY" in off_df.columns else off_df.get("PLAY_TYPE", pd.Series(dtype=str))
            tmp = pd.DataFrame({"concepts": txtcol.apply(_infer_concepts), "SUCCESS": off_df.get("SUCCESS")}).explode("concepts")
            concept_success = tmp.groupby("concepts")["SUCCESS"].mean().to_dict()
        else:
            concept_success = {}
        def _score_row(r: pd.Series) -> float:
            score = 0.0
            cov_tags = str(r.get('COVERAGE_TAGS','')).lower()
            if 'vs man' in cov_tags: score += 1.2 * cov_dist.get('MAN', 0)
            if ('vs c3' in cov_tags) or ('vs cover 3' in cov_tags): score += 1.2 * cov_dist.get('C3', 0)
            if any(k in cov_tags for k in ['vs quarters','vs c4','vs cover 4']): score += 1.2 * cov_dist.get('C4', 0)
            if any(k in cov_tags for k in ['vs cover 2','vs c2']): score += 1.2 * cov_dist.get('C2', 0)
            if opp_blitz3 >= 0.35 and 'vs blitz' in str(r.get('PRESSURE_TAGS','')).lower(): score += 0.4
            cons = str(r.get('CONCEPT_TAGS','')).lower(); hits = [k for k,kws in concept_map.items() if any(w in cons for w in kws)]
            if hits: score += float(np.mean([concept_success.get(k, 0.5) for k in hits]))
            return float(score)
        lib_scored = lib.copy(); lib_scored["__SCORE__"] = lib_scored.apply(_score_row, axis=1)
        def _pick_top(df: pd.DataFrame, keywords: List[str], limit: int = 6) -> pd.DataFrame:
            def _contains(row):
                hay = (str(row.get('CONCEPT_TAGS','')) + ' ' + str(row.get('SITUATION_TAGS','')) + ' ' + str(row.get('PLAY_NAME',''))).lower();
                return any(k in hay for k in keywords)
            cand = df[df.apply(_contains, axis=1)].co
            cand = cand.sort_values("__SCORE__", ascending=False)
            picked, used_names, used_forms = [], set(), set()
            for _, rr in cand.iterrows():
                name, form = rr.get('PLAY_NAME'), rr.get('FORMATION')
                if name in used_names or form in used_forms: continue
                picked.append(rr); used_names.add(name); used_forms.add(form)
                if len(picked) >= limit: break
