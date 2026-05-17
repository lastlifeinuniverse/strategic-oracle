"""
Strategic Oracle  Streamlit Chat Interface
============================================
Clean, readable UI with NUS project capability showcase.

Run with:
    streamlit run oracle_app.py
"""

# Fix google namespace conflict  must happen before streamlit import
import sys
import os

try:
    import google.genai
    import google.genai.types
except Exception:
    pass

import streamlit as st
import glob
import queue
import threading

# Ensure imports resolve relative to this file's folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import oracle backend
try:
    from strategic_oracle_gemini import (
        OracleServices,
        OracleDB,
        build_graph,
        OracleState,
        HARDCODED_PROFILE,
        MAX_HISTORY,
        IMAGE_OUTPUT_DIR,
        wrap_with_logging,
    )
    ORACLE_IMPORT_ERROR = None
except Exception as _import_err:
    import traceback
    ORACLE_IMPORT_ERROR = traceback.format_exc()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Strategic Oracle",
    page_icon="[TAO]",
    layout="wide",
    initial_sidebar_state="expanded",
)

if ORACLE_IMPORT_ERROR:
    st.error("[FAIL] Failed to import strategic_oracle_gemini")
    st.code(ORACLE_IMPORT_ERROR, language="python")
    st.stop()

# =============================================================================
# CSS  clean warm light theme, readable, NUS-appropriate
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #f5f0e8;
    color: #2c2416;
}
[data-testid="stSidebar"] {
    background-color: #faf6ee;
    border-right: 2px solid #e8dcc8;
}
[data-testid="stSidebar"] * { color: #2c2416 !important; }

.sidebar-section {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8b6914 !important;
    margin: 14px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #e8dcc8;
}
.info-card {
    background: #ffffff;
    border: 1px solid #e8dcc8;
    border-left: 3px solid #c9a84c;
    border-radius: 6px;
    padding: 9px 12px;
    margin-bottom: 7px;
}
.info-card .label {
    font-size: 0.63rem;
    color: #8b7355;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 2px;
}
.info-card .value {
    font-family: 'Noto Serif', serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #2c2416;
}
.info-card .desc {
    font-size: 0.74rem;
    color: #6b5a3e;
    margin-top: 2px;
    line-height: 1.4;
}
.weather-card {
    background: linear-gradient(135deg, #e8f4fd, #dceefa);
    border: 1px solid #b8d4e8;
    border-left: 3px solid #4a90c4;
    border-radius: 6px;
    padding: 9px 12px;
    margin-bottom: 7px;
}
.weather-card .value { color: #1a3a5c; font-family:'Noto Serif',serif; font-weight:600; }
.weather-card .desc  { color: #2a5a8c; font-size:0.74rem; margin-top:3px; line-height:1.6; }

.cap-badge {
    display: inline-block;
    background: #fdf6e3;
    border: 1px solid #c9a84c;
    color: #8b6914;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 20px;
    margin-bottom: 4px;
    margin-top: 8px;
}

.stButton > button {
    background-color: #ffffff !important;
    color: #2c2416 !important;
    border: 1px solid #d4c4a0 !important;
    border-radius: 6px !important;
    font-size: 0.77rem !important;
    padding: 6px 10px !important;
    text-align: left !important;
    line-height: 1.4 !important;
    white-space: normal !important;
    height: auto !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background-color: #fdf6e3 !important;
    border-color: #c9a84c !important;
    color: #8b6914 !important;
}

[data-testid="stChatMessage"] {
    background-color: #ffffff;
    border: 1px solid #e8dcc8;
    border-radius: 10px;
    margin-bottom: 10px;
}
[data-testid="stChatMessage"] p {
    color: #2c2416 !important;
    line-height: 1.75;
}
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #2c2416 !important;
    border: 2px solid #d4c4a0 !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #c9a84c !important;
}
hr { border-color: #e8dcc8 !important; margin: 10px 0 !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# INIT
# =============================================================================
@st.cache_resource()
def init_oracle():
    svc      = OracleServices()
    db       = OracleDB("oracle.db")
    compiled = build_graph(svc, db)
    graph    = wrap_with_logging(compiled, db)
    return svc, db, graph

for _k, _v in [("messages", []), ("history", []), ("meta", None),
                ("weather", None), ("context_loaded", False),
                ("profile_source", "dict"), ("profile_label", "Default (hardcoded profile)"),
                ("profile_name", HARDCODED_PROFILE['name']),
                ("profile_occupation", HARDCODED_PROFILE['occupation'])]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

try:
    svc, db, graph = init_oracle()
except Exception as _e:
    st.error(f"[FAIL] Oracle initialisation failed: {_e}")
    st.stop()

# Ensure profile_source exists on svc (guard against stale cache)
if not hasattr(svc, "profile_source"):
    svc.profile_source = "dict"

if not st.session_state.context_loaded:
    with st.spinner("Loading Bazi and weather data..."):
        st.session_state.meta    = svc.get_today_meta()
        st.session_state.weather = svc.get_weather()
        st.session_state.context_loaded = True
        if getattr(svc, 'profile_source', 'dict') == "pdf":
            st.session_state.profile_source = "pdf"
            st.session_state.profile_label  = "profile.pdf (auto-loaded)" 

meta    = st.session_state.meta
weather = st.session_state.weather


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 4px 0;">
        <div style="font-family:'Noto Serif',serif; font-size:1.35rem; color:#2c2416; letter-spacing:3px;">
            [TAO] STRATEGIC ORACLE
        </div>
        <div style="font-size:0.68rem; color:#8b7355; letter-spacing:1.5px; margin-top:3px;">
            HEALTH  BAZI  METAPHYSICS  AI ADVISOR
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # =========================================================================
    # TIER 1: HEALTH MONITORING (Garmin — most used)
    # =========================================================================
    st.markdown('<div class="sidebar-section" style="color:#c9a84c;">HEALTH MONITORING</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">[GARMIN] Training & Recovery</div>', unsafe_allow_html=True)

    garmin_count = db.get_garmin_activity_count()
    garmin_col   = "#4caf50" if garmin_count > 0 else "#bbb"
    st.markdown(
        f'<div class="info-card" style="border-left-color:{garmin_col};">'
        f'<div class="label">Synced Activities</div>'
        f'<div class="value" style="font-size:0.9rem;">{garmin_count} record{"s" if garmin_count != 1 else ""}</div>'
        f'<div class="desc">Zone analysis, VO2 max, training optimization</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if "show_garmin" not in st.session_state:
        st.session_state.show_garmin = False
    if "garmin_status" not in st.session_state:
        st.session_state.garmin_status = None
    if "garmin_mfa_pending" not in st.session_state:
        st.session_state.garmin_mfa_pending = False

    # Credential helpers — hoisted so re-sync button can use them
    import json as _json
    _CRED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".garmin_saved.json")

    def _load_saved_creds():
        try:
            if os.path.exists(_CRED_FILE):
                data = _json.load(open(_CRED_FILE))
                return data.get("email", ""), data.get("password", "")
        except Exception:
            pass
        return "", ""

    def _load_saved_email():
        return _load_saved_creds()[0]

    def _save_creds(email: str, password: str = ""):
        try:
            existing = {}
            if os.path.exists(_CRED_FILE):
                try:
                    existing = _json.load(open(_CRED_FILE))
                except Exception:
                    pass
            existing["email"] = email
            if password:
                existing["password"] = password
            _json.dump(existing, open(_CRED_FILE, "w"))
        except Exception:
            pass

    # ── Re-sync button (FIRST — most used daily action) ──────────────────
    saved_email_qs, saved_password_qs = _load_saved_creds()
    from garmin_sync import TOKENSTORE as _TSTORE
    _tokens_ok = (
        saved_email_qs
        and os.path.isdir(_TSTORE)
        and any(os.scandir(_TSTORE))
    )
    if _tokens_ok:
        if st.button("🔄 Re-sync Garmin (get latest activity)", use_container_width=True, key="btn_garmin_resync"):
            with st.spinner("Syncing latest Garmin data..."):
                try:
                    from garmin_sync import GarminSync
                    _gs = GarminSync()
                    _ok, _msg = _gs.connect(saved_email_qs, saved_password_qs, mfa_code="")
                    if _ok:
                        _acts = _gs.get_recent_activities(days=14)
                        _stats = _gs.get_today_stats()
                        _comp = _gs.get_comprehensive_health_data()
                        try:
                            from garmin_sync import get_personalized_zones
                            _zones = get_personalized_zones(db.db_path)
                            if _zones.get("vt1_bpm"):
                                _acts = _gs.enrich_activities_with_zones(_acts, _zones)
                        except Exception as _ze:
                            print(f"[resync] zone enrichment error: {_ze}")
                        _n = db.import_garmin_activities(_acts)
                        if _stats:
                            db.upsert_garmin_daily_stats(_stats)
                        if _comp:
                            db.upsert_garmin_comprehensive_health(_comp)
                        st.session_state.garmin_status = ("ok", f"Re-synced — {_n} new activities, zone data updated.")
                        st.rerun()
                    else:
                        st.session_state.garmin_status = ("warn", f"Re-sync failed: {_msg}")
                        st.rerun()
                except Exception as _exc:
                    st.session_state.garmin_status = ("err", f"Re-sync error: {_exc}")
                    st.rerun()

    # ── Quick analysis buttons (compact row) ─────────────────────────────
    _gc1, _gc2, _gc3 = st.columns(3)
    with _gc1:
        if st.button("🩺 Today", use_container_width=True, key="quick_garmin_today"):
            st.session_state.pending_query = (
                "How am I doing today? Focus on: "
                "1) Yesterday's recovery — sleep quality and stages, HRV, resting HR, breathing rate, body battery at wake, stress score. "
                "2) Today's activities — if I have logged any activity today, analyse it: was the intensity appropriate, how does it affect my recovery, what's the cumulative load? "
                "Give a GREEN/AMBER/RED verdict for today and a specific prescription for what I should still do or avoid for the rest of today."
            )
            st.rerun()
    with _gc2:
        if st.button("📅 This Wk", use_container_width=True, key="quick_garmin_week_review"):
            st.session_state.pending_query = (
                "Give me a full weekly review of how I fared this week — current calendar week from Monday up to now. "
                "Include: all activities logged this week with dates, duration, and heart rate, "
                "zone distribution and training load, recovery metrics (HRV, sleep, resting HR, body battery) across the week, "
                "weekly intensity minutes vs WHO targets, and an overall verdict on whether my training this week was optimal. "
                "End with specific recommendations for the coming days."
            )
            st.rerun()
    with _gc3:
        if st.button("📊 Last Wk", use_container_width=True, key="quick_garmin_week"):
            st.session_state.pending_query = "Give me a full review of last week — all activities logged, zone distribution, recovery metrics (HRV, sleep, body battery), and training recommendations going forward."
            st.rerun()

    # ── Daily Recovery & Bazi Guidance (dedicated card) ─────────────────────
    st.markdown('<div class="sidebar-section" style="margin-top: 12px;">Daily Guidance</div>', unsafe_allow_html=True)

    if st.button("💙 Recovery & Bazi", use_container_width=True, key="daily_recovery_bazi"):
        st.session_state.show_daily_recovery_bazi = not st.session_state.get("show_daily_recovery_bazi", False)
        st.rerun()

    if st.session_state.get("show_daily_recovery_bazi"):
        with st.container(border=True):
            st.markdown("### 🔮 Today's Recovery & Bazi Guidance")

            # Fetch recovery data
            recovery_data = db.get_daily_recovery_snapshot()

            # Fetch Bazi meta
            from strategic_oracle_gemini import OracleServices
            svc_temp = OracleServices()
            meta = svc_temp.get_today_meta()

            if not recovery_data.get("has_data"):
                st.warning(
                    "⏳ **Recovery data not synced yet**\n\n"
                    "Your sleep, HRV, body battery, and resting HR data need to sync from Garmin.\n\n"
                    "**Next steps:**\n"
                    "1. Click 🔄 RE-SYNC GARMIN (button above)\n"
                    "2. Complete the login and sync\n"
                    "3. Wait 15–30 minutes for your Garmin device to send data\n"
                    "4. Refresh this card\n\n"
                    "**But meanwhile**, based on today's Bazi element alone:"
                )
            else:
                # Display recovery metrics
                daily = recovery_data.get("daily_stats", {})
                comp = recovery_data.get("comprehensive", {})

                col1, col2 = st.columns(2)
                with col1:
                    if daily.get("sleep_hours"):
                        st.metric("Sleep", f"{daily['sleep_hours']}h", f"Score: {daily.get('sleep_score', '?')}")
                    if daily.get("body_battery"):
                        st.metric("Body Battery", f"{daily['body_battery']}%")
                with col2:
                    if daily.get("resting_hr"):
                        st.metric("Resting HR", f"{daily['resting_hr']} bpm")
                    if comp.get("hrv_last_night_avg"):
                        st.metric("HRV", f"{comp['hrv_last_night_avg']} ms", f"({comp.get('hrv_status', '?')})")

                st.divider()

            # Display Bazi guidance (always available) — in compass format
            st.markdown("**🧭 Bazi Compass (Rule-based)**")

            col_focus, col_watch = st.columns(2)
            with col_focus:
                st.markdown(f"**Focus:** {meta.get('do_today', '—')}")
            with col_watch:
                st.markdown(f"**Watch:** {meta.get('psyche_note', '—')}")

            st.markdown(f"*{meta.get('day_quality', 'Balanced')}*")

            # Expandable detailed Bazi breakdown
            with st.expander("📖 Full Bazi Breakdown", expanded=False):
                st.write(f"**Day Quality:** {meta.get('day_quality', 'Unknown')}")
                st.write(f"**Personal Impact:** {meta.get('personal_impact', '—')}")
                st.write(f"**Officer Energy:** {meta.get('officer_energy', '—')} ({meta.get('officer_personal', '—')})")
                st.write(f"**Branch Energy:** {meta.get('branch_note', '—')}")
                st.write(f"**Avoid:** {meta.get('avoid_today', '—')}")
                st.info(f"🎯 {meta.get('training_advice', 'Connect with your Fire element through purposeful movement.')}")

    # ── Divination Reflection (Claude Opus 4.7) ──────────────────────────
    if st.button("✨ Divination Reflection", use_container_width=True, key="divination_btn"):
        st.session_state.show_divination = not st.session_state.get("show_divination", False)
        st.rerun()

    if st.session_state.get("show_divination"):
        with st.container(border=True):
            st.markdown("### ✨ Daily Divination Reflection")
            st.caption("Psychologically-grounded archetypal reflection — not fortune telling.")

            # Show current life context with edit option
            current_ctx = db.get_current_life_context()

            with st.expander("📝 Current Life Context (used in reflection)", expanded=False):
                with st.form("life_context_form", clear_on_submit=False):
                    new_situation = st.text_area(
                        "Life Situation",
                        value=current_ctx.get("situation", ""),
                        height=70
                    )
                    new_emotional = st.text_area(
                        "Emotional State",
                        value=current_ctx.get("emotional_state", ""),
                        height=70
                    )
                    new_focus = st.text_area(
                        "Current Focus",
                        value=current_ctx.get("focus_area", ""),
                        height=70
                    )
                    new_events = st.text_area(
                        "Recent Events",
                        value=current_ctx.get("recent_events", ""),
                        height=70
                    )
                    new_phase = st.text_input(
                        "Life Phase",
                        value=current_ctx.get("phase", "")
                    )
                    new_goals = st.text_area(
                        "Current Goals",
                        value=current_ctx.get("goals", ""),
                        height=70
                    )
                    if st.form_submit_button("💾 Save Context", use_container_width=True):
                        db.save_life_context({
                            "situation":       new_situation,
                            "emotional_state": new_emotional,
                            "focus_area":      new_focus,
                            "recent_events":   new_events,
                            "phase":           new_phase,
                            "goals":           new_goals,
                        })
                        st.success("Context updated.")
                        st.rerun()

            # Generation controls
            today_existing = db.get_today_divination()
            col_gen, col_model = st.columns([2, 1])
            with col_gen:
                gen_label = "🔄 Regenerate" if today_existing else "✨ Generate"
                gen_clicked = st.button(gen_label, use_container_width=True, key="div_generate")
            with col_model:
                model_choice = st.selectbox(
                    "Model",
                    options=["deepseek", "gemini"],
                    index=0,
                    key="div_model_choice",
                    label_visibility="collapsed"
                )

            if gen_clicked:
                spinner_msg = f"Weaving reflection via {model_choice.title()}..."
                with st.spinner(spinner_msg):
                    try:
                        from divination_module import DivinationService
                        svc = DivinationService(db=db, model_choice=model_choice)
                        result = svc.generate_daily_reflection()
                        if result["success"]:
                            st.session_state.last_divination = result
                            st.rerun()
                        else:
                            st.error(f"Generation failed: {result.get('error', 'unknown error')}")
                    except Exception as e:
                        st.error(f"Error: {e}")

            # Get display result (cached or freshly generated)
            display_result = st.session_state.get("last_divination") or today_existing

            if display_result:
                st.divider()

                # Display compact 3-field compass
                st.markdown("**🧭 Today's Compass**")

                col_focus, col_watch = st.columns(2)
                with col_focus:
                    st.markdown(f"**Focus:** {display_result.get('focus', '—')}")
                with col_watch:
                    st.markdown(f"**Watch:** {display_result.get('watch', '—')}")

                # Display headline
                st.markdown(f"*{display_result.get('headline', '—')}*")

                st.divider()

                # Expandable full reflection
                with st.expander("📖 Full 8-Part Reflection", expanded=False):
                    st.markdown(display_result.get("full_reflection", "No full reflection available."))

                # Metadata caption
                cap = f"Model: {display_result.get('model', '?')} | {display_result.get('timestamp', '')[:10]}"
                if display_result.get("from_cache"):
                    cap += " | (cached)"
                st.caption(cap)

    # ── Garmin status messages ────────────────────────────────────────────
    if st.session_state.garmin_status:
        kind, txt = st.session_state.garmin_status
        if kind == "ok":
            st.success(txt)
        elif kind == "warn":
            st.warning(txt)
        else:
            st.error(txt)

    # ── Import / Connect (rarely needed — at bottom of Garmin section) ──
    if st.button("[GARMIN] Import Garmin Data", use_container_width=True, key="btn_garmin"):
        st.session_state.show_garmin = not st.session_state.show_garmin
        st.session_state.garmin_mfa_pending = False

    if st.session_state.show_garmin:
        tab_api, tab_csv = st.tabs(["🔗 Connect to Garmin", "📁 CSV Upload"])

        #  Tab 1: API login (default)
        with tab_api:
            def _save_email(email: str):
                _save_creds(email, "")

            from garmin_sync import TOKENSTORE
            saved_email, saved_password = _load_saved_creds()
            # Tokens exist if the TOKENSTORE directory has any files in it
            tokens_exist = (
                saved_email
                and os.path.isdir(TOKENSTORE)
                and any(os.scandir(TOKENSTORE))
            )

            #  Quick-connect path: tokens cached, just press Connect
            if tokens_exist and not st.session_state.garmin_mfa_pending:
                st.success(f"Saved account: **{saved_email}**")
                st.caption("OAuth tokens cached — no password needed.")
                col1, col2 = st.columns([2, 1])
                with col1:
                    quick_btn = st.button("🔄 Connect & Sync", use_container_width=True, key="btn_garmin_quick")
                with col2:
                    if st.button("Change", use_container_width=True, key="btn_garmin_change"):
                        _save_creds("", "")
                        st.rerun()

                if quick_btn:
                    with st.spinner("Connecting with saved account..."):
                        try:
                            from garmin_sync import GarminSync
                            gs = GarminSync()
                            ok, msg = gs.connect(saved_email, saved_password, mfa_code="")
                            if ok:
                                activities  = gs.get_recent_activities(days=14)
                                today_stats = gs.get_today_stats()
                                comprehensive_health = gs.get_comprehensive_health_data()
                                try:
                                    from garmin_sync import get_personalized_zones
                                    _cz = get_personalized_zones(db.db_path)
                                    if _cz.get("vt1_bpm"):
                                        activities = gs.enrich_activities_with_zones(activities, _cz)
                                except Exception as _ze:
                                    print(f"[sync] zone enrichment error: {_ze}")
                                n = db.import_garmin_activities(activities)
                                if today_stats:
                                    db.upsert_garmin_daily_stats(today_stats)
                                if comprehensive_health:
                                    db.upsert_garmin_comprehensive_health(comprehensive_health)
                                status_msg = f"Synced — {n} new activities, daily stats updated"
                                if comprehensive_health:
                                    status_msg += ", comprehensive health data synced"
                                status_msg += "."
                                st.session_state.garmin_status = ("ok", status_msg)
                                st.session_state.show_garmin = False
                                st.rerun()
                            else:
                                # Tokens expired — fall back to password login
                                _save_creds(saved_email, "")
                                st.session_state.garmin_status = (
                                    "warn", "Session expired — please log in again."
                                )
                                st.rerun()
                        except Exception as exc:
                            st.session_state.garmin_status = ("err", f"[FAIL] Sync error: {exc}")

            else:
                #  Full login path
                st.caption("Credentials are saved locally for quick reconnect next time.")
                g_email = st.text_input(
                    "Garmin email", value=saved_email, key="g_email"
                )
                g_pass = st.text_input(
                    "Garmin password", value=saved_password, type="password", key="g_pass"
                )

                if st.session_state.garmin_mfa_pending:
                    st.info("2FA enabled — enter the code from your authenticator app.")
                    g_mfa = st.text_input("MFA / 2FA code", key="g_mfa", max_chars=8)
                else:
                    g_mfa = ""

                if st.button("🔗 Connect & Sync", use_container_width=True, key="btn_garmin_go"):
                    if g_email and g_pass:
                        with st.spinner("Connecting to Garmin Connect..."):
                            try:
                                from garmin_sync import GarminSync
                                gs = GarminSync()
                                ok, msg = gs.connect(g_email, g_pass, mfa_code=g_mfa)

                                if msg == "MFA_REQUIRED":
                                    st.session_state.garmin_mfa_pending = True
                                    _save_creds(g_email, g_pass)
                                    st.session_state.garmin_status = (
                                        "warn",
                                        "2FA required — enter your MFA code above and click Connect & Sync again.",
                                    )
                                    st.rerun()
                                elif ok:
                                    _save_creds(g_email, g_pass)
                                    activities  = gs.get_recent_activities(days=7)
                                    today_stats = gs.get_today_stats()
                                    comprehensive_health = gs.get_comprehensive_health_data()
                                    n = db.import_garmin_activities(activities)
                                    if today_stats:
                                        db.upsert_garmin_daily_stats(today_stats)
                                    if comprehensive_health:
                                        db.upsert_garmin_comprehensive_health(comprehensive_health)
                                    status_msg = f"Connected — {n} new activities imported, daily stats updated"
                                    if comprehensive_health:
                                        status_msg += ", comprehensive health data synced"
                                    status_msg += "."
                                    st.session_state.garmin_status = ("ok", status_msg)
                                    st.session_state.garmin_mfa_pending = False
                                    st.session_state.show_garmin = False
                                    st.rerun()
                                else:
                                    st.session_state.garmin_status = ("err", f"[FAIL] {msg}")
                            except Exception as exc:
                                st.session_state.garmin_status = ("err", f"[FAIL] Sync error: {exc}")
                    else:
                        st.warning("Enter both email and password.")

        #  Tab 2: CSV upload (alternative method)
        with tab_csv:
            st.caption(
                "In Garmin Connect → Activities → menu (top right) → Export CSV. "
                "Upload the downloaded file here."
            )
            uploaded_csv = st.file_uploader(
                "Choose Garmin activities CSV",
                type=["csv"],
                key="garmin_csv",
                label_visibility="collapsed",
            )
            if uploaded_csv and st.button("Import CSV", use_container_width=True, key="btn_csv_go"):
                with st.spinner("Parsing CSV..."):
                    try:
                        from garmin_sync import parse_garmin_csv
                        activities, err = parse_garmin_csv(uploaded_csv.read())
                        if err:
                            st.session_state.garmin_status = ("err", f"[FAIL] {err}")
                        else:
                            n = db.import_garmin_activities(activities)
                            st.session_state.garmin_status = (
                                "ok",
                                f"Imported {n} new activities from {uploaded_csv.name}.",
                            )
                            st.session_state.show_garmin = False
                            st.rerun()
                    except Exception as exc:
                        st.session_state.garmin_status = ("err", f"[FAIL] Parse error: {exc}")

    # =========================================================================
    # TIER 2: DAILY GUIDANCE (Bazi, Weather)
    # =========================================================================
    st.divider()
    st.markdown('<div class="sidebar-section" style="color:#c9a84c;">DAILY GUIDANCE</div>', unsafe_allow_html=True)

    #  Bazi — Personalised Daily Divination
    if meta:
        _q_col = meta.get('quality_colour', '#c9a84c')
        _quality = meta.get('day_quality', 'Balanced')
        st.markdown(f'<div class="sidebar-section">[BAZI] Today\'s Divination — {meta["day_pillar"]}</div>', unsafe_allow_html=True)

        # Day Quality Banner
        st.markdown(f"""
        <div class="info-card" style="border-left-color:{_q_col};border-left-width:4px;">
            <div class="label">Day Quality for Your Yang Fire</div>
            <div class="value" style="color:{_q_col};">{_quality}</div>
            <div class="desc">{meta.get('personal_impact', meta['element_desc'])}</div>
        </div>
        """, unsafe_allow_html=True)

        # Officer Impact
        _off_energy = meta.get('officer_energy', '')
        st.markdown(f"""
        <div class="info-card">
            <div class="label">{meta['officer_name']} Officer — {_off_energy} Energy</div>
            <div class="value" style="font-size:0.85rem;">{meta.get('officer_personal', meta['officer_meaning'])}</div>
            <div class="desc">{meta.get('officer_advice', '')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Do / Avoid / Training
        _do = meta.get('do_today', '')
        _avoid = meta.get('avoid_today', '')
        _training = meta.get('training_advice', '')
        _psyche = meta.get('psyche_note', '')
        _branch = meta.get('branch_note', '')

        if _do or _avoid:
            st.markdown(f"""
            <div class="info-card" style="border-left-color:#2e7d32;">
                <div class="label">Do Today</div>
                <div class="desc" style="color:#1b5e20;">{_do}</div>
            </div>
            <div class="info-card" style="border-left-color:#c62828;">
                <div class="label">Avoid Today</div>
                <div class="desc" style="color:#b71c1c;">{_avoid}</div>
            </div>
            """, unsafe_allow_html=True)

        if _training:
            st.markdown(f"""
            <div class="info-card" style="border-left-color:#1565c0;">
                <div class="label">Training Guidance</div>
                <div class="desc" style="color:#0d47a1;">{_training}</div>
            </div>
            """, unsafe_allow_html=True)

        if _psyche:
            st.markdown(f"""
            <div class="info-card" style="border-left-color:#6a1b9a;">
                <div class="label">Mind & Psyche</div>
                <div class="desc" style="color:#4a148c;">{_psyche}</div>
            </div>
            """, unsafe_allow_html=True)

        # Branch + Sector (compact)
        st.markdown(f"""
        <div class="info-card">
            <div class="label">Branch Energy — {meta.get('branch_energy', '')}</div>
            <div class="desc">{_branch}</div>
            <div class="label" style="margin-top:6px;">Auspicious Sector</div>
            <div class="value">{meta['auspicious_sector']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Quick buttons for Bazi
    if meta:
        bazi_col1, bazi_col2 = st.columns(2)
        with bazi_col1:
            if st.button("🎋 Deep Dive", use_container_width=True, key="quick_bazi_meaning"):
                st.session_state.pending_query = "Give me a deep reading of today's Bazi — how does the day pillar, officer, and elemental energy specifically interact with my Yang Fire Day Master? What should I prioritise and what should I be cautious about?"
                st.rerun()
        with bazi_col2:
            if st.button("⚠️ What to Avoid", use_container_width=True, key="quick_bazi_avoid"):
                st.session_state.pending_query = "Based on today's Bazi energy and my chart, what specific situations, decisions, or behaviours should I avoid today? Be practical and specific."
                st.rerun()

    #  Weather
    if weather:
        st.markdown('<div class="sidebar-section">[WEATHER] Live Weather  Singapore</div>', unsafe_allow_html=True)
        rain_icon = "" if weather.get("is_rainy") else ""
        outdoor   = "[OK] Suitable for outdoors" if weather.get("is_outdoor_safe") else "[WARNING] Recommend indoors"
        st.markdown(f"""
        <div class="weather-card">
            <div class="value">{rain_icon} {weather['condition']}</div>
            <div class="desc">
                 {weather['temp_c']}C (feels {weather['feels_like']}C)<br>
                 Humidity {weather['humidity']}%  UV Index {weather['uv_index']}<br>
                {outdoor}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Quick buttons for Weather
    if weather:
        weather_col1, weather_col2, weather_col3 = st.columns(3)
        with weather_col1:
            if st.button("🌅 Outdoors?", use_container_width=True, key="quick_weather_outdoors"):
                st.session_state.pending_query = "Is it a good day to go outdoors?"
                st.rerun()
        with weather_col2:
            if st.button("🏃 Activity", use_container_width=True, key="quick_weather_activity"):
                st.session_state.pending_query = "What activity suits the current weather?"
                st.rerun()
        with weather_col3:
            if st.button("⏰ Best Time", use_container_width=True, key="quick_weather_time"):
                st.session_state.pending_query = "What's the best time to go outside?"
                st.rerun()

    # =========================================================================
    # KNOWLEDGE BASE (Profile, RAG, Lab, DB)
    # =========================================================================
    st.divider()
    st.markdown('<div class="sidebar-section" style="color:#c9a84c;">KNOWLEDGE BASE</div>', unsafe_allow_html=True)

    #  User Profile
    st.markdown('<div class="sidebar-section">[USER] Profile</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-card">'
        f'<div class="label">Name</div>'
        f'<div class="value">{st.session_state.profile_name}</div>'
        f'<div class="desc">{st.session_state.profile_occupation}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    #  PDF Profile Status
    st.markdown('<div class="sidebar-section">[DOC] RAG Profile Source</div>', unsafe_allow_html=True)

    is_pdf   = st.session_state.profile_source == "pdf"
    src_icon = "[DOC]" if is_pdf else "[FORM]"
    src_col  = "#4caf50" if is_pdf else "#c9a84c"
    src_bg   = "#f1f8f1" if is_pdf else "#fffbf0"
    src_text = "#1b5e20" if is_pdf else "#7b5800"
    st.markdown(
        f'<div class="info-card" style="border-left-color:{src_col};background:{src_bg};">'        f'<div class="label">Active source</div>'        f'<div class="value" style="font-size:0.82rem;color:{src_text};">{src_icon} {st.session_state.profile_label}</div>'        f'<div class="desc">Used by the RAG agent to answer profile questions</div>'        f'</div>',
        unsafe_allow_html=True
    )

    # Upload button  reveals file picker
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(" Upload PDF", use_container_width=True, key="btn_upload"):
            st.session_state.show_uploader = not st.session_state.show_uploader
    with col_b:
        if is_pdf:
            if st.button(" Use Default", use_container_width=True, key="btn_revert"):
                svc.chunks = svc._build_chunks_from_profile_dict()
                svc.profile_source = "dict"
                st.session_state.profile_source    = "dict"
                st.session_state.profile_label     = "Default (hardcoded profile)"
                st.session_state.profile_name      = HARDCODED_PROFILE['name']
                st.session_state.profile_occupation = HARDCODED_PROFILE['occupation']
                st.session_state.show_uploader  = False
                st.session_state.messages = []
                st.session_state.history  = []
                st.rerun()

    if st.session_state.show_uploader:
        uploaded_pdf = st.file_uploader(
            "Choose a profile PDF",
            type=["pdf"],
            help="The PDF will be chunked and used for all RAG Q&A queries.",
            key="profile_uploader",
            label_visibility="collapsed"
        )
        if uploaded_pdf is not None:
            if st.button("[OK] Load Profile", use_container_width=True, key="btn_load"):
                pdf_bytes = uploaded_pdf.read()
                success, msg, pages = svc.load_profile_from_pdf(pdf_bytes)
                if success:
                    st.session_state.profile_source = "pdf"
                    st.session_state.profile_label  = f"{uploaded_pdf.name} ({pages}p)"
                    st.session_state.show_uploader  = False
                    st.session_state.messages = []
                    st.session_state.history  = []
                    # Extract display name from PDF filename as best guess
                    # (e.g. mock_profile_lim_wei_ling.pdf  Lim Wei Ling)
                    _raw = uploaded_pdf.name.replace(".pdf","").replace("mock_profile_","")
                    _raw = _raw.replace("profile_","").replace("_"," ").replace("-"," ")
                    _pdf_name = _raw.title().strip() or "Uploaded Profile"
                    _pdf_occ  = f"Profile loaded from {uploaded_pdf.name}"
                    # Try to find name in first chunk of PDF text
                    if svc.chunks:
                        import re as _re
                        _chunk = svc.chunks[0][:300]
                        _nm = _re.search(r"(?:Name|User)[:\s]+([A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+)", _chunk)
                        if _nm:
                            _pdf_name = _nm.group(1).strip()
                        _om = _re.search(r"(?:Occupation|Role|Position)[:\s]+(.{5,80})", _chunk)
                        if _om:
                            _pdf_occ = _om.group(1).strip()[:80]
                    st.session_state.profile_name       = _pdf_name
                    st.session_state.profile_occupation = _pdf_occ
                    st.rerun()
                else:
                    st.error(f"[FAIL] {msg}")

    #  SQLite Memory Status
    st.markdown('<div class="sidebar-section">[DB] SQLite Memory</div>', unsafe_allow_html=True)
    stats = db.get_stats()
    _conv_count = stats.get("conversation_log", 0)
    _conv_col = "#4caf50" if _conv_count > 0 else "#bbb"
    st.markdown(
        f'<div class="info-card" style="border-left-color:{_conv_col};">'
        f'<div class="label">Conversations</div>'
        f'<div class="value" style="font-size:0.9rem;">{_conv_count} record{"s" if _conv_count != 1 else ""}</div>'
        f'<div class="desc">Every query + Oracle response stored with timestamp</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    #  Lab Reports 
    st.markdown('<div class="sidebar-section">[LAB] Lab Reports</div>', unsafe_allow_html=True)

    lab_summary = db.get_lab_summary()
    lab_col = "#4caf50" if lab_summary["has_data"] else "#bbb"
    abnormal_count = len(lab_summary.get("abnormal", []))
    abnormal_txt = f"  {abnormal_count} flagged" if abnormal_count else ""
    st.markdown(
        f'<div class="info-card" style="border-left-color:{lab_col};">'
        f'<div class="label">Uploaded Reports</div>'
        f'<div class="value" style="font-size:0.9rem;">'
        f'{lab_summary["report_count"]} report{"s" if lab_summary["report_count"] != 1 else ""}'
        f'  {lab_summary["marker_count"]} markers{abnormal_txt}</div>'
        f'<div class="desc">Innoquest  VO2 Master  SPOT-MAS  Longevity</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if "show_lab_upload" not in st.session_state:
        st.session_state.show_lab_upload = False
    if "lab_status" not in st.session_state:
        st.session_state.lab_status = None

    # VO2 Master GXT data — always show so it can be re-seeded if needed
    _VO2_MARKERS = [
        ("VO2_MAX",           30.0,  "mL/kg/min"),
        ("VO2_MAX_PERCENTILE",  5.0,  "%ile"),
        ("MAX_HR_MEASURED",   185.0,  "bpm"),
        ("MAX_POWER",         340.0,  "W"),
        ("VT1_HR",            156.0,  "bpm"),
        ("VT2_HR",            173.0,  "bpm"),
        ("VT1_POWER",         169.0,  "W"),
        ("VT2_POWER",         259.0,  "W"),
        ("GXT_ZONE1_HR_MAX",  140.0,  "bpm"),
        ("GXT_ZONE2_HR_MAX",  156.0,  "bpm"),
        ("GXT_ZONE3_HR_MAX",  164.0,  "bpm"),
        ("GXT_ZONE4_HR_MAX",  173.0,  "bpm"),
    ]
    if st.button("⚡ Load VO2 Test Data (15 Oct 2025)", use_container_width=True, key="btn_seed_vo2"):
        n = db.seed_vo2_master_data(_VO2_MARKERS, "2025-10-15", "VO2 Master GXT — Chen Yingkai")
        st.success(f"VO2 Master data loaded — {n} markers written. Zones now active.")
        st.rerun()

    if st.button("[FORM] Upload Lab Report", use_container_width=True, key="btn_lab"):
        st.session_state.show_lab_upload = not st.session_state.show_lab_upload

    if st.session_state.show_lab_upload:
        uploaded_lab = st.file_uploader(
            "Choose a lab report PDF",
            type=["pdf"],
            key="lab_pdf",
            label_visibility="collapsed",
        )
        if uploaded_lab and st.button(" Parse & Import", use_container_width=True,
                                       key="btn_lab_go"):
            with st.spinner("Extracting lab markers..."):
                try:
                    from lab_parser import parse_lab_pdf
                    pdf_bytes = uploaded_lab.read()
                    report, err = parse_lab_pdf(pdf_bytes, gemini_fn=svc.gemini,
                                                gemini_vision_fn=svc.gemini_vision)
                    if err:
                        st.session_state.lab_status = ("err", f"[FAIL] {err}")
                    else:
                        db.save_lab_report(report, pdf_name=uploaded_lab.name)
                        flags = {}
                        for m in report.markers:
                            flags[m.flag] = flags.get(m.flag, 0) + 1
                        flag_str = "  ".join(f"{v} {k}" for k, v in flags.items())
                        st.session_state.lab_status = (
                            "ok",
                            f"Imported {len(report.markers)} markers from "
                            f"{report.source} ({report.collected_at})  {flag_str}",
                        )
                        st.session_state.show_lab_upload = False
                        st.rerun()
                except Exception as exc:
                    st.session_state.lab_status = ("err", f"[FAIL] Parse error: {exc}")

    if st.session_state.lab_status:
        kind, txt = st.session_state.lab_status
        if kind == "ok":
            st.success(txt)
        else:
            st.error(txt)

    st.divider()

    #  Oracle Queries — focused on health + metaphysics
    st.markdown('<div class="sidebar-section">[ORACLE] Quick Queries</div>', unsafe_allow_html=True)

    CAPABILITIES = [
        {
            "badge": "Health & Training",
            "queries": [
                "What activity should I do today based on my recovery and Bazi?",
                "Based on my training data, what should I focus on this week?",
            ]
        },
        {
            "badge": "Bazi & Metaphysics",
            "queries": [
                "Tell me about my Bazi chart and personality traits.",
                "What are my goals and life direction?",
            ]
        },
        {
            "badge": "Conversational Memory",
            "queries": [
                "What did we talk about earlier?",
                "Based on what I shared, what should I focus on today?",
            ]
        },
        {
            "badge": "Weather & Activity",
            "queries": [
                "Is it a good day to go outdoors?",
                "Suggest an activity suited to the current weather.",
            ]
        },
    ]

    for cap in CAPABILITIES:
        st.markdown(f'<div class="cap-badge">{cap["badge"]}</div>', unsafe_allow_html=True)
        for q in cap["queries"]:
            if st.button(q, key=f"cap_{q[:35]}", use_container_width=True):
                st.session_state["pending_query"] = q

    st.divider()

    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()


# =============================================================================
# MAIN CHAT AREA
# =============================================================================
st.markdown("""
<div style="text-align:center; padding:1.2rem 0 1rem 0; border-bottom:2px solid #e8dcc8; margin-bottom:1.5rem;">
    <div style="font-family:'Noto Serif',serif; font-size:1.8rem; color:#2c2416; letter-spacing:3px;">
        [TAO] STRATEGIC ORACLE
    </div>
    <div style="font-size:0.78rem; color:#8b7355; letter-spacing:1px; margin-top:4px;">
        Health Monitoring &nbsp;&nbsp; Bazi Metaphysics &nbsp;&nbsp; Chinese Metaphysics &nbsp;&nbsp; AI Advisor
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome card on first load
if not st.session_state.messages:
    day_pillar = meta['day_pillar'] if meta else ""
    officer    = meta['officer_name'] if meta else ""
    condition  = weather['condition'] if weather else ""
    temp       = weather['temp_c'] if weather else ""
    _quality   = meta.get('day_quality', 'Balanced') if meta else "Balanced"
    _q_col     = meta.get('quality_colour', '#c9a84c') if meta else "#c9a84c"
    _impact    = meta.get('personal_impact', '') if meta else ""
    _off_energy = meta.get('officer_energy', '') if meta else ""
    _do        = meta.get('do_today', '') if meta else ""
    _first_name = ' '.join(st.session_state.profile_name.split()[1:]) or st.session_state.profile_name.split()[0]
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #fff9ee, #fdf4e3);
        border: 1px solid #e8dcc8;
        border-left: 4px solid {_q_col};
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 20px;
    ">
        <div style="font-family:'Noto Serif',serif; font-size:1.05rem; color:#2c2416; margin-bottom:8px;">
            Today's divination for {_first_name} — <strong style="color:{_q_col};">{_quality}</strong>
        </div>
        <div style="font-size:0.87rem; color:#6b5a3e; line-height:1.9;">
            <strong>{day_pillar}</strong> pillar day under <strong>{_off_energy}</strong> ({officer}) energy.<br>
            {_impact}<br>
            <span style="color:#1b5e20;"><strong>Do:</strong> {_do}</span><br>
            Singapore: <strong>{condition}, {temp}C</strong>.
        </div>
        <div style="margin-top:10px; font-size:0.76rem; color:#8b7355;">
            Ask me about your health, training, Bazi guidance, or what to do and avoid today.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    avatar = "✨" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        for img_path in msg.get("images", []):
            if os.path.exists(img_path):
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
            else:
                st.caption(f" _{os.path.basename(img_path)} (file not found)_")


# =============================================================================
# ORACLE INVOKE
# =============================================================================
def find_new_images(before: set, since_time: float = None) -> list:
    """Return images created after `before` snapshot or after `since_time` timestamp."""
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    # Small settle delay to ensure file writes are flushed to disk
    import time as _t
    _t.sleep(0.3)
    all_imgs = glob.glob(os.path.join(IMAGE_OUTPUT_DIR, "*.png"))
    if since_time:
        # Also catch by modification time  handles thread timing issues
        new_by_time = [f for f in all_imgs if os.path.getmtime(f) >= since_time - 1.0]
        new_by_set  = [f for f in all_imgs if f not in before]
        combined    = sorted(set(new_by_time) | set(new_by_set))
        return combined
    return sorted(set(all_imgs) - before)


def run_oracle(query: str, cancel_event: threading.Event = None, history_snapshot: list = None):
    import time as _t
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    existing  = set(glob.glob(os.path.join(IMAGE_OUTPUT_DIR, "*.png")))
    t_start      = _t.time()
    wall_start   = _t.time()  # Wall clock for mtime comparison

    _img_words = ["generate image", "with image", "visualise", "visualize",
                  "show image", "illustrate", "generate a picture", "draw me",
                  "create an image", "make an image", "show me an image",
                  "with an image", "and show", "show image"]
    wants_img = any(w in query.lower() for w in _img_words)

    if history_snapshot is None:
        history_snapshot = []
    history_snapshot = list(history_snapshot)
    history_snapshot.append({"role": "user", "content": query})

    state = {
        "query":          query,
        "intent":         None,
        "meta":           None,
        "weather":        None,
        "history":        history_snapshot,
        "response":       None,
        "want_images":    wants_img,
        "garmin_context": None,
        "lab_context":    None,
        "cancel_event":   cancel_event,
    }

    # Run graph.invoke in a daemon thread with a result queue
    result_q = queue.Queue()

    def _worker():
        try:
            result_q.put(("ok", graph.invoke(state)))
        except Exception as e:
            result_q.put(("err", e))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    # Poll so cancel_event can interrupt the wait
    while t.is_alive():
        t.join(timeout=0.5)
        if cancel_event and cancel_event.is_set():
            return "*Request cancelled.*", [], _t.time() - t_start, {}

    if not result_q.empty():
        status, payload = result_q.get()
    else:
        return "[WARNING] The Oracle timed out. Please try again.", [], 120.0

    if status == "err":
        raise payload

    result   = payload
    elapsed  = _t.time() - t_start
    response = result.get("response", "I wasn't able to process that. Please try rephrasing.")
    new_imgs = find_new_images(existing, since_time=wall_start)
    # Return result dict so caller can update session state on the main thread
    return response, new_imgs, elapsed, result


# =============================================================================
# INPUT
# =============================================================================
import time as _tw

# Ensure required session state keys always exist
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle in-progress oracle request across reruns
if st.session_state.get("_oracle_running"):
    worker       = st.session_state["_oracle_worker"]
    result_holder = st.session_state["_oracle_result"]
    cancel_event  = st.session_state["_cancel_event"]
    user_input    = st.session_state["_oracle_query"]

    with st.chat_message("assistant", avatar="✨"):
        _col1, _col2 = st.columns([8, 1])
        _col1.markdown("*Consulting the Oracle...*")
        if _col2.button("⏹", key="oracle_cancel", help="Cancel"):
            cancel_event.set()

    if worker.is_alive():
        _tw.sleep(0.4)
        st.rerun()
    else:
        st.session_state["_oracle_running"] = False
        status, payload = result_holder.get() if not result_holder.empty() else ("ok", ("*Request cancelled.*", [], 0.0, {}))
        try:
            if status == "err":
                raise payload
            response, new_images, elapsed, result = payload
        except Exception as e:
            response, new_images, elapsed, result = f"[WARNING] Error: {e}\n\nPlease try rephrasing.", [], 0.0, {}

        # Update history on main thread (safe)
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history = result.get("history", st.session_state.history)
        if len(st.session_state.history) > MAX_HISTORY:
            st.session_state.history = st.session_state.history[-MAX_HISTORY:]

        with st.chat_message("assistant", avatar="✨"):
            st.markdown(response)
            if new_images:
                for img_path in new_images:
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f" {os.path.basename(img_path)}", use_container_width=True)
            st.caption(f" Response time: {elapsed:.1f}s")

        st.session_state.messages.append({"role": "assistant", "content": response, "images": new_images})

pending    = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask the Oracle...") or pending

if user_input and not st.session_state.get("_oracle_running"):
    st.session_state.messages.append({"role": "user", "content": user_input, "images": []})

    # Snapshot history on main thread before handing off to worker thread
    history_snap  = list(st.session_state.history)
    cancel_event  = threading.Event()
    result_holder = queue.Queue()

    def _run_and_store():
        try:
            result_holder.put(("ok", run_oracle(user_input, cancel_event, history_snap)))
        except Exception as e:
            result_holder.put(("err", e))

    worker = threading.Thread(target=_run_and_store, daemon=True)
    st.session_state["_oracle_running"] = True
    st.session_state["_oracle_worker"]  = worker
    st.session_state["_oracle_result"]  = result_holder
    st.session_state["_cancel_event"]   = cancel_event
    st.session_state["_oracle_query"]   = user_input
    worker.start()
    st.rerun()

elif user_input and st.session_state.get("_oracle_running"):
    st.warning("Please wait for the current request to finish.")