# app.py
import os
import csv
from io import StringIO
from datetime import datetime, timezone, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from openai import OpenAI

# Optional: saubere Exception-Typen (je nach SDK-Version verfügbar)
try:
    from openai import AuthenticationError, RateLimitError, APIError, BadRequestError
except Exception:  # pragma: no cover
    AuthenticationError = RateLimitError = APIError = BadRequestError = Exception


# -----------------------------
# Basics
# -----------------------------
st.set_page_config(page_title="AI-CFO Firmenversion", layout="wide")

EXPECTED_COLS = ["month", "revenue", "fixed_costs", "variable_costs", "employees", "cash"]
NUM_COLS = ["revenue", "fixed_costs", "variable_costs", "employees", "cash"]

AUDIT_FILE = "audit_log.csv"
MAX_PER_60MIN = 3


# -----------------------------
# Helpers
# -----------------------------
def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_audit_header():
    if not os.path.exists(AUDIT_FILE):
        try:
            with open(AUDIT_FILE, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "user_name", "model", "rows_count", "result", "error_type"])
        except Exception:
            # Nicht crashen, nur später warnen
            pass


def audit_log(user_name: str, model: str, rows_count: int, result: str, error_type: str):
    ensure_audit_header()
    try:
        with open(AUDIT_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([now_iso_utc(), user_name, model, rows_count, result, error_type])
    except Exception:
        st.warning("Audit-Log konnte nicht geschrieben werden (Dateisystem nicht verfügbar).")


def read_audit_df() -> pd.DataFrame:
    if not os.path.exists(AUDIT_FILE):
        return pd.DataFrame(columns=["timestamp", "user_name", "model", "rows_count", "result", "error_type"])
    try:
        return pd.read_csv(AUDIT_FILE)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "user_name", "model", "rows_count", "result", "error_type"])


def rate_limit_ok(user_name: str) -> (bool, int):
    """Max. 3 Analysen pro 60 Minuten pro user_name (gezählt über Audit-Log)."""
    df = read_audit_df()
    if df.empty or "timestamp" not in df.columns:
        return True, 0

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    except Exception:
        return True, 0

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=60)
    recent = df[(df["user_name"] == user_name) & (df["timestamp"] >= cutoff)]
    used = len(recent)  # zählt OK + ERROR als Versuch
    return used < MAX_PER_60MIN, used


def parse_csv_text(text: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(text))


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Spalten trimmen
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        st.error(f"CSV hat fehlende Spalten: {', '.join(missing)}")
        st.stop()

    df = df[EXPECTED_COLS]

    # month immer als Text speichern (Data Editor ist dann stabil)
    df["month"] = df["month"].astype(str).str.strip()

    # Numerische Spalten
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Hinweis bei NaNs
    if df[NUM_COLS].isna().any().any():
        st.warning("Hinweis: Einige Zahlen konnten nicht gelesen werden (leer/ungültig). Bitte im Data Editor korrigieren.")

    # Sortierung: wenn month wie Datum aussieht, sortiere danach
    month_dt = pd.to_datetime(df["month"], errors="coerce")
    if month_dt.notna().mean() >= 0.6:
        df["_month_dt"] = month_dt
        df = df.sort_values(by="_month_dt", ascending=True).drop(columns=["_month_dt"])
    else:
        df = df.sort_values(by="month", ascending=True)

    df = df.reset_index(drop=True)

    if len(df) == 0:
        st.error("Keine Datenzeilen gefunden.")
        st.stop()

    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    profit = df["revenue"] - (df["fixed_costs"] + df["variable_costs"])
    avg_profit = profit.mean(skipna=True)

    last = df.iloc[-1]
    last_cash = float(last["cash"]) if pd.notna(last["cash"]) else float("nan")

    # Runway
    if pd.notna(avg_profit) and avg_profit > 0:
        runway = "∞"
    else:
        burn = float(-avg_profit) if pd.notna(avg_profit) else float("nan")
        if burn and burn > 0 and pd.notna(last_cash):
            runway = f"{(last_cash / burn):.1f}"
        else:
            runway = "n/a"

    # Kostenquote (letzter Monat)
    last_rev = float(last["revenue"]) if pd.notna(last["revenue"]) else float("nan")
    if pd.notna(last["fixed_costs"]) and pd.notna(last["variable_costs"]):
        last_costs = float(last["fixed_costs"] + last["variable_costs"])
    else:
        last_costs = float("nan")

    if pd.notna(last_rev) and last_rev != 0 and pd.notna(last_costs):
        cost_ratio = last_costs / last_rev
    else:
        cost_ratio = float("nan")

    # Umsatz pro Mitarbeiter (letzter Monat)
    last_emp = float(last["employees"]) if pd.notna(last["employees"]) else float("nan")
    if pd.notna(last_emp) and last_emp != 0 and pd.notna(last_rev):
        rev_per_emp = last_rev / last_emp
    else:
        rev_per_emp = float("nan")

    # Umsatzwachstum (erstes -> letztes)
    first_rev = float(df.iloc[0]["revenue"]) if pd.notna(df.iloc[0]["revenue"]) else float("nan")
    if pd.notna(first_rev) and first_rev != 0 and pd.notna(last_rev):
        rev_growth = (last_rev - first_rev) / first_rev
    else:
        rev_growth = float("nan")

    return {
        "avg_profit": avg_profit,
        "last_cash": last_cash,
        "runway": runway,
        "cost_ratio": cost_ratio,
        "rev_per_emp": rev_per_emp,
        "rev_growth": rev_growth,
        "profit_series": profit,
        "total_costs_series": (df["fixed_costs"] + df["variable_costs"]),
    }


def fmt_money(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:,.0f}".replace(",", ".")


def fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x*100:.1f}%".replace(".", ",")


def month_to_label(m):
    return str(m)


def build_prompt(df: pd.DataFrame, k: dict) -> str:
    last_n = df.tail(6).copy()
    csv_preview = last_n.to_csv(index=False)

    return f"""
Du bist ein erfahrener CFO-Berater. Antworte auf Deutsch, klar und direkt.

Ziel: Erzeuge eine Management-Empfehlung basierend auf den Finanzdaten (Monate).

KPIs (aus den Daten berechnet):
- Ø Monatsgewinn: {k["avg_profit"]}
- Cash (letzter Monat): {k["last_cash"]}
- Runway (Monate): {k["runway"]}
- Kostenquote (letzter Monat, Kosten/Umsatz): {k["cost_ratio"]}
- Umsatz pro Mitarbeiter (letzter Monat): {k["rev_per_emp"]}
- Umsatzwachstum (erstes -> letztes Monat): {k["rev_growth"]}

Daten (letzte 6 Monate als CSV):
{csv_preview}

Pflicht-Ausgabe (genau diese Struktur, bitte einhalten):
1) Executive Summary (max. 5 Sätze)
2) Runway-Einschätzung
3) Hiring-Empfehlung (JA/NEIN/WANN + Begründung)
4) Kostenstrategie
5) 3 konkrete Management-Entscheidungen (nummeriert 1-3)
6) 2 Risiken (Bulletpoints)
""".strip()


# -----------------------------
# Sidebar: Login + Settings
# -----------------------------
st.sidebar.title("Zugang")

app_pw = os.getenv("APP_PASSWORD", "")
user_name = st.sidebar.text_input("Benutzername", value=st.session_state.get("user_name", ""))
pw_in = st.sidebar.text_input("Kurs-Passwort", type="password")

if not app_pw:
    st.error("APP_PASSWORD fehlt. Bitte Server-Secret/Umgebungsvariable APP_PASSWORD setzen. App ist gesperrt.")
    st.stop()

auth_ok = bool(user_name.strip()) and (pw_in == app_pw)

if not auth_ok:
    st.info("Bitte einloggen (Benutzername + Kurs-Passwort).")
    st.stop()

st.session_state["user_name"] = user_name.strip()

st.sidebar.divider()
model = st.sidebar.selectbox("Modell", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
st.sidebar.caption("Hinweis: OpenAI-Key wird nur serverseitig gelesen (OPENAI_API_KEY).")


# -----------------------------
# Main UI
# -----------------------------
st.title("AI-CFO Dashboard – Firmenversion (Streamlit)")

tab1, tab2 = st.tabs(["CSV Upload", "Manuell bearbeiten"])

default_csv = """month,revenue,fixed_costs,variable_costs,employees,cash
2025-08,120000,40000,30000,8,250000
2025-09,130000,40000,32000,8,265000
2025-10,125000,41000,31000,9,272000
2025-11,140000,42000,33000,9,285000
2025-12,150000,43000,36000,10,295000
2026-01,155000,44000,37000,10,310000
"""

df_raw = None

with tab1:
    up = st.file_uploader("CSV hochladen", type=["csv"])
    if up is not None:
        try:
            df_raw = pd.read_csv(up)
        except Exception:
            st.error("CSV konnte nicht gelesen werden. Bitte Dateiformat prüfen.")
            st.stop()

with tab2:
    st.write("CSV in das Textfeld einfügen (inkl. Kopfzeile) und laden:")
    csv_text = st.text_area("CSV-Text", value=st.session_state.get("csv_text", default_csv), height=180)
    colA, colB = st.columns([1, 2])
    with colA:
        load_text = st.button("Aus Text laden")
    with colB:
        st.caption("Tipp: Danach im Data Editor Zahlen/Monate korrigieren.")
    if load_text:
        st.session_state["csv_text"] = csv_text
        try:
            df_raw = parse_csv_text(csv_text)
        except Exception:
            st.error("CSV-Text konnte nicht gelesen werden. Bitte Trennzeichen/Zeilen prüfen.")
            st.stop()

# Wenn Upload benutzt wurde, überschreibt es den Text
if df_raw is None:
    if "csv_text" in st.session_state:
        try:
            df_raw = parse_csv_text(st.session_state["csv_text"])
        except Exception:
            df_raw = parse_csv_text(default_csv)
    else:
        df_raw = parse_csv_text(default_csv)

df = validate_and_clean(df_raw)

st.subheader("Daten (prüfen & bearbeiten)")
edited = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "month": st.column_config.TextColumn("month"),
        "revenue": st.column_config.NumberColumn("revenue"),
        "fixed_costs": st.column_config.NumberColumn("fixed_costs"),
        "variable_costs": st.column_config.NumberColumn("variable_costs"),
        "employees": st.column_config.NumberColumn("employees"),
        "cash": st.column_config.NumberColumn("cash"),
    },
)
df = validate_and_clean(pd.DataFrame(edited))

# KPIs
k = compute_kpis(df)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Ø Monatsgewinn", fmt_money(k["avg_profit"]))
c2.metric("Cash (letzter Monat)", fmt_money(k["last_cash"]))
c3.metric("Runway (Monate)", str(k["runway"]))
c4.metric("Kostenquote (letzter Monat)", fmt_pct(k["cost_ratio"]))
c5.metric("Umsatz/Mitarbeiter (letzter Monat)", fmt_money(k["rev_per_emp"]))
c6.metric("Umsatzwachstum", fmt_pct(k["rev_growth"]))

# Chart
st.subheader("Trend (Umsatz, Gesamtkosten, Gewinn)")
chart_df = df.copy()
chart_df["month_label"] = chart_df["month"].apply(month_to_label)
chart_df["total_costs"] = k["total_costs_series"]
chart_df["profit"] = k["profit_series"]

fig = plt.figure(figsize=(10, 3))
plt.plot(chart_df["month_label"], chart_df["revenue"], label="Umsatz")
plt.plot(chart_df["month_label"], chart_df["total_costs"], label="Gesamtkosten")
plt.plot(chart_df["month_label"], chart_df["profit"], label="Gewinn")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# Button: GPT Auswertung
st.subheader("KI-Empfehlung")
st.caption("Wichtig: Ohne gültigen OPENAI_API_KEY (Secret) keine Auswertung möglich.")

if st.button("Empfehlung berechnen", type="primary"):
    uname = st.session_state.get("user_name", "unknown")
    rows_count = int(len(df))

    # Rate limit (vor API)
    ok, used = rate_limit_ok(uname)
    if not ok:
        st.error(f"Rate-Limit erreicht: max. {MAX_PER_60MIN} Analysen pro 60 Minuten. (Aktuell: {used})")
        audit_log(uname, model, rows_count, "ERROR", "other")
        st.stop()

    # Key nur serverseitig
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key.strip():
        st.error('OpenAI API Key fehlt. Bitte Server-Secret OPENAI_API_KEY setzen. Ohne Key keine Auswertung möglich.')
        audit_log(uname, model, rows_count, "ERROR", "other")
        st.stop()

    prompt = build_prompt(df, k)

    try:
        client = OpenAI(api_key=api_key)

        resp = client.responses.create(
            model=model,
            input=prompt,
        )

        out = getattr(resp, "output_text", None)
        if not out:
            out = "Keine Textausgabe erhalten (resp.output_text leer)."

        st.success("Auswertung erstellt.")
        st.markdown(out)

        audit_log(uname, model, rows_count, "OK", "none")

    except AuthenticationError:
        st.error("OpenAI Fehler (401): Key ungültig/fehlende Berechtigung. Ohne gültigen Key keine Auswertung möglich.")
        audit_log(uname, model, rows_count, "ERROR", "401")
    except RateLimitError:
        st.error("OpenAI Fehler (429): Rate Limit erreicht. Ohne gültigen Key/Quota keine Auswertung möglich.")
        audit_log(uname, model, rows_count, "ERROR", "429")
    except (BadRequestError, APIError):
        st.error("OpenAI Fehler: Anfrage fehlgeschlagen. Ohne gültigen Key keine Auswertung möglich.")
        audit_log(uname, model, rows_count, "ERROR", "other")
    except Exception:
        st.error("OpenAI Fehler: Unbekannter Fehler. Ohne gültigen Key keine Auswertung möglich.")
        audit_log(uname, model, rows_count, "ERROR", "other")

with st.expander("Audit-Log anzeigen (optional)"):
    adf = read_audit_df()
    st.dataframe(adf, use_container_width=True)
