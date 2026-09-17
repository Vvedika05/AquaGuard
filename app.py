
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# AquaGuard
# Real-Time Water Safety Monitoring & Classification System
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
META_PATH = ROOT / "outputs" / "model_config.json"
EDA_DIR = ROOT / "outputs" / "eda"
EVAL_DIR = ROOT / "outputs" / "evaluation"

st.set_page_config(
    page_title="AquaGuard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
     <style>
    /* ============================================================
   Global text visibility fix
   ============================================================ */


/* Main application text */
.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div,
.stApp li,
.stApp td,
.stApp th {
    color: #000000;
}

/* Streamlit headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #000000 !important;
}

/* Sidebar text */
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #000000 !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6 {
    color: #000000 !important;
}

/* Captions */
.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #000000 !important;
}

/* Radio buttons */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] span {
    color: #000000 !important;
}

/* Select boxes / input labels */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    color: #000000 !important;
}

/* Metric labels and values */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div,
div[data-testid="stMetric"] p {
    color: #000000 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #000000 !important;
}

/* Expanders */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {
    color: #000000 !important;
}

/* Dataframe text */
[data-testid="stDataFrame"] {
    color: #000000 !important;
}

/* General markdown text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong {
    color: #000000;
}
   
        :root {
            --ag-navy: #12304A;
            --ag-deep: #0B2438;
            --ag-teal: #0F9D8A;
            --ag-aqua: #21B7C6;
            --ag-sky: #E6F7FA;
            --ag-mint: #E9F8F3;
            --ag-red: #D95C5C;
            --ag-red-bg: #FFF0EF;
            --ag-ink: #243746;
            --ag-muted: #5F7480;
            --ag-line: #C9E0E4;
            --ag-panel: #F9FDFD;
            --ag-soft: #EDF7F8;
        }

        /* Soft water-tinted application background */
        .stApp {
            background:
                radial-gradient(circle at 7% 2%, rgba(33, 183, 198, 0.16), transparent 28%),
                radial-gradient(circle at 94% 10%, rgba(15, 157, 138, 0.13), transparent 27%),
                radial-gradient(circle at 50% 100%, rgba(77, 126, 168, 0.08), transparent 30%),
                linear-gradient(135deg, #DFF4F7 0%, #EEF9F7 48%, #E1F1F5 100%);
        }

        /* Main content sits on a lightly translucent panel rather than plain white */
        .main .block-container {
            background: rgba(246, 251, 251, 0.88);
            border-radius: 18px;
            margin-top: 0.7rem;
            padding-left: 2rem;
            padding-right: 2rem;
            box-shadow: 0 8px 35px rgba(18, 48, 74, 0.07);
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.45rem;
            padding-bottom: 3rem;
        }

        .app-title {
            font-size: 2.45rem;
            font-weight: 760;
            color: var(--ag-navy);
            margin-bottom: 0.12rem;
            letter-spacing: -0.035em;
        }

        .app-subtitle {
            color: var(--ag-muted);
            font-size: 1.02rem;
            margin-bottom: 1.35rem;
        }

        .section-title {
            font-size: 1.30rem;
            font-weight: 720;
            color: var(--ag-navy);
            margin-top: 1.45rem;
            margin-bottom: 0.65rem;
        }
        .section-title::before {
            content: "";
            display: inline-block;
            width: 5px;
            height: 20px;
            margin-right: 9px;
            vertical-align: -2px;
            border-radius: 4px;
            background: linear-gradient(180deg, var(--ag-aqua), var(--ag-teal));
        }


        .muted {
            color: var(--ag-muted);
            font-size: 0.88rem;
        }

        .info-strip {
            border: 1px solid #CFE8EC;
            border-left: 4px solid var(--ag-aqua);
            border-radius: 8px;
            padding: 0.82rem 1rem;
            background: var(--ag-sky);
            color: var(--ag-ink);
        }

        .status-card {
            border-radius: 10px;
            padding: 1.05rem 1.3rem;
            border: 1px solid var(--ag-line);
            margin: 0.5rem 0 0.8rem 0;
            box-shadow: 0 2px 10px rgba(18, 48, 74, 0.045);
        }

        .status-card.potable {
            background: var(--ag-mint);
            border-left: 7px solid var(--ag-teal);
        }

        .status-card.nonpotable {
            background: var(--ag-red-bg);
            border-left: 7px solid var(--ag-red);
        }

        .status-label {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            color: var(--ag-muted);
            margin-bottom: 0.20rem;
        }

        .status-value {
            font-size: 1.75rem;
            font-weight: 780;
            color: var(--ag-navy);
        }

        .meter-wrap {
            margin: 0.65rem 0 1.1rem 0;
        }

        .meter-label-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.35rem;
            color: #526570;
            font-size: 0.88rem;
        }

        .meter {
            width: 100%;
            height: 16px;
            background: #E6EEF0;
            border-radius: 9px;
            overflow: hidden;
            border: 1px solid #D5E1E4;
        }

        .meter-fill {
            height: 100%;
            border-radius: 9px;
            transition: width 0.25s ease;
        }

        .reading-box {
            border: 1px solid var(--ag-line);
            border-top: 3px solid var(--ag-aqua);
            border-radius: 8px;
            padding: 0.72rem 0.85rem;
            background: var(--ag-panel);
            min-height: 78px;
            margin-bottom: 0.65rem;
            box-shadow: 0 1px 7px rgba(18, 48, 74, 0.035);
        }

        .reading-name {
            color: var(--ag-muted);
            font-size: 0.77rem;
            margin-bottom: 0.22rem;
        }

        .reading-value {
            color: var(--ag-navy);
            font-size: 1.08rem;
            font-weight: 700;
        }

        .live-header {
            border: 1px solid #D1E9EC;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            background: linear-gradient(90deg, #F0FBFC 0%, #F7FBFA 100%);
            margin-bottom: 1rem;
        }

        .live-header-title {
            color: var(--ag-navy);
            font-size: 1.05rem;
            font-weight: 720;
        }

        .live-header-sub {
            color: var(--ag-muted);
            font-size: 0.84rem;
            margin-top: 0.15rem;
        }

        .mode-note {
            border: 1px solid var(--ag-line);
            border-radius: 8px;
            padding: 0.72rem 0.9rem;
            background: #FAFCFC;
            color: #536671;
            font-size: 0.88rem;
            margin-bottom: 0.9rem;
        }

        .small-status {
            display: inline-block;
            border-radius: 20px;
            padding: 0.26rem 0.68rem;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            border: 1px solid #CFE3E7;
            background: var(--ag-sky);
            color: #24717B;
            margin-bottom: 0.25rem;
        }

        .history-title {
            color: var(--ag-navy);
            font-size: 1.05rem;
            font-weight: 720;
            margin-bottom: 0.35rem;
        }
        .hero-panel {
            position: relative;
            overflow: hidden;
            border: 1px solid #CDE8EB;
            border-radius: 14px;
            padding: 1.55rem 1.65rem;
            margin: 0.2rem 0 1.25rem 0;
            background: linear-gradient(135deg, #D6F1F4 0%, #E8F8F4 58%, #DCEFF3 100%);
            box-shadow: 0 5px 20px rgba(18, 48, 74, 0.055);
        }

        .hero-panel:after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            right: -65px;
            top: -85px;
            border-radius: 50%;
            background: rgba(33, 183, 198, 0.10);
        }

        .hero-kicker {
            color: #24717B;
            font-size: 0.76rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
        }

        .hero-heading {
            color: var(--ag-navy);
            font-size: 1.72rem;
            font-weight: 780;
            line-height: 1.18;
            margin-bottom: 0.45rem;
            max-width: 850px;
        }

        .hero-copy {
            color: #5A6F7C;
            font-size: 0.96rem;
            line-height: 1.55;
            max-width: 900px;
        }

        .objective-card {
            min-height: 180px;
            border: 1px solid var(--ag-line);
            border-radius: 11px;
            padding: 1.05rem 1.05rem 0.9rem 1.05rem;
            background: rgba(252, 255, 255, 0.92);
            box-shadow: 0 4px 15px rgba(18, 48, 74, 0.055);
        }

        .objective-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: var(--ag-sky);
            color: #24717B;
            font-size: 0.78rem;
            font-weight: 780;
            margin-bottom: 0.65rem;
        }

        .objective-title {
            color: var(--ag-navy);
            font-size: 1.02rem;
            font-weight: 740;
            margin-bottom: 0.38rem;
        }

        .objective-copy {
            color: #687984;
            font-size: 0.86rem;
            line-height: 1.48;
        }

        .procedure-card {
            min-height: 168px;
            border: 1px solid var(--ag-line);
            border-radius: 11px;
            padding: 1rem;
            background: #FFFFFF;
            box-shadow: 0 3px 12px rgba(18, 48, 74, 0.04);
        }

        .procedure-step {
            color: #24717B;
            font-size: 0.73rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }

        .procedure-title {
            color: var(--ag-navy);
            font-size: 1rem;
            font-weight: 740;
            margin-bottom: 0.35rem;
        }

        .procedure-copy {
            color: #687984;
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .phase-card {
            min-height: 205px;
            border-radius: 12px;
            border: 1px solid var(--ag-line);
            padding: 1.05rem;
            background: rgba(252, 255, 255, 0.92);
            box-shadow: 0 4px 15px rgba(18, 48, 74, 0.055);
        }

        .phase-tag {
            display: inline-block;
            border-radius: 18px;
            padding: 0.24rem 0.58rem;
            background: var(--ag-sky);
            color: #24717B;
            font-size: 0.72rem;
            font-weight: 760;
            margin-bottom: 0.55rem;
        }

        .phase-title {
            color: var(--ag-navy);
            font-size: 1.05rem;
            font-weight: 750;
            margin-bottom: 0.42rem;
        }

        .phase-copy {
            color: #687984;
            font-size: 0.85rem;
            line-height: 1.48;
        }

        .phase-list {
            color: #4F6572;
            font-size: 0.82rem;
            line-height: 1.55;
            padding-left: 1.05rem;
            margin: 0.45rem 0 0 0;
        }

        .callout {
            border-radius: 10px;
            border: 1px solid #D5E8EA;
            border-left: 5px solid var(--ag-teal);
            padding: 0.9rem 1rem;
            background: #F4FBF9;
            color: #4F6572;
            line-height: 1.5;
        }

        .mini-flow {
            display: flex;
            align-items: stretch;
            gap: 0.5rem;
            margin: 0.35rem 0 1.25rem 0;
        }

        .mini-flow-item {
            flex: 1;
            border: 1px solid var(--ag-line);
            border-radius: 9px;
            padding: 0.75rem 0.7rem;
            background: #FFFFFF;
        }

        .mini-flow-num {
            color: #24717B;
            font-size: 0.70rem;
            font-weight: 780;
            letter-spacing: 0.08em;
        }

        .mini-flow-title {
            color: var(--ag-navy);
            font-size: 0.88rem;
            font-weight: 720;
            margin-top: 0.18rem;
        }

        .mini-flow-arrow {
            display: flex;
            align-items: center;
            color: #9AB1BA;
            font-weight: 700;
        }

        .objective-card:nth-child(1) { border-top: 4px solid #21B7C6; }
        .objective-card:nth-child(2) { border-top: 4px solid #0F9D8A; }
        .objective-card:nth-child(3) { border-top: 4px solid #4D7EA8; }

        .phase-card:nth-child(1) { border-top: 4px solid #21B7C6; }
        .phase-card:nth-child(2) { border-top: 4px solid #0F9D8A; }
        .phase-card:nth-child(3) { border-top: 4px solid #4D7EA8; }
        .phase-card:nth-child(4) { border-top: 4px solid #D99A45; }

        @media (max-width: 900px) {
            .mini-flow { flex-wrap: wrap; }
            .mini-flow-arrow { display: none; }
            .mini-flow-item { min-width: 45%; }
        }


        div[data-testid="stMetric"] {
            border: 1px solid var(--ag-line);
            border-radius: 8px;
            padding: 0.62rem 0.8rem;
            background: rgba(248, 253, 253, 0.94);
            box-shadow: 0 2px 9px rgba(18, 48, 74, 0.04);
        }

        div[data-testid="stButton"] > button {
            border-radius: 7px;
            border: 1px solid #C8DDE1;
            font-weight: 650;
        }

        div[data-testid="stButton"] > button[kind="primary"] {
            background: var(--ag-navy);
            border-color: var(--ag-navy);
            color: white;
        }

        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: #1A4768;
            border-color: #1A4768;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid #C8E0E4;
            background: linear-gradient(180deg, #E6F6F8 0%, #EEF8F6 58%, #E7F4F6 100%);
        }

        [data-testid="stSidebar"] > div:first-child {
            background: transparent;
        }

        [data-testid="stSidebar"] h2 {
            color: var(--ag-navy);
        }

        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        button[data-baseweb="tab"] {
            font-weight: 650;
        }

        div[data-baseweb="tab-highlight"] {
            background: var(--ag-teal);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Load model + metadata
# -----------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {META_PATH}")

    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


try:
    model, meta = load_artifacts()
except Exception as exc:
    st.error(
        "AquaGuard could not load its trained model or metadata. "
        "Run `python src\\train_models.py` from the project root first."
    )
    st.exception(exc)
    st.stop()


FEATURES = meta["input_features"]
RANGES = meta["feature_ranges"]
TARGET = meta["target_column"]
SELECTED = meta["selected_model"]


# -----------------------------
# Helpers
# -----------------------------
def pct(value):
    return "—" if value is None else f"{value:.1%}"


def predict_sample(sample):
    frame = pd.DataFrame([sample])
    prediction = int(model.predict(frame)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0, 1])

    return prediction, probability


def prediction_label(prediction):
    return "POTABLE" if prediction == 1 else "NON-POTABLE"


def prediction_color(prediction):
    return "#15966b" if prediction == 1 else "#d64545"


def prediction_card(prediction, probability, title="MODEL PREDICTION"):
    label = prediction_label(prediction)
    cls = "potable" if prediction == 1 else "nonpotable"

    st.markdown(
        f"""
        <div class="status-card {cls}">
            <div class="status-label">{title}</div>
            <div class="status-value">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probability is not None:
        st.markdown(
            f"""
            <div class="meter-wrap">
                <div class="meter-label-row">
                    <span>Model probability of Potable</span>
                    <strong>{probability:.1%}</strong>
                </div>
                <div class="meter">
                    <div class="meter-fill"
                         style="width:{probability * 100:.1f}%;
                                background:{prediction_color(prediction)};">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def feature_table(sample):
    rows = []

    for feature in FEATURES:
        value = float(sample[feature])
        info = RANGES.get(feature, {})

        if "min" in info:
            minimum = info["min"]
            maximum = info["max"]
            inside = minimum <= value <= maximum
            range_text = f"{minimum:.3f} – {maximum:.3f}"
            status = "Within observed range" if inside else "Outside observed range"
        else:
            range_text = "—"
            status = "Categorical value"

        rows.append(
            {
                "Parameter": feature,
                "Value": round(value, 4),
                "Observed training range": range_text,
                "Status": status,
            }
        )

    return pd.DataFrame(rows)


def outside_training_range(sample):
    result = []

    for feature in FEATURES:
        info = RANGES.get(feature, {})
        if "min" not in info:
            continue

        value = float(sample[feature])
        if value < info["min"] or value > info["max"]:
            result.append(
                {
                    "Parameter": feature,
                    "Input": value,
                    "Observed minimum": info["min"],
                    "Observed maximum": info["max"],
                }
            )

    return pd.DataFrame(result)


def make_baseline():
    """
    Use the median of each observed training feature as the monitoring
    starting point. This is a representative data-derived baseline,
    not a recommended water-quality profile.
    """
    baseline = {}

    for feature in FEATURES:
        info = RANGES.get(feature, {})

        if "median" in info:
            baseline[feature] = float(info["median"])
        elif "categories" in info and info["categories"]:
            baseline[feature] = info["categories"][0]
        else:
            baseline[feature] = 0.0

    return baseline


def move_toward_contamination(sample):
    """
    Change readings using the observed training-data distribution.
    No prediction label is hard-coded. The SVM reclassifies the result.
    """
    updated = dict(sample)

    for feature in FEATURES:
        info = RANGES.get(feature, {})

        if "min" not in info:
            continue

        span = max(info["p95"] - info["p05"], 1e-9)
        value = float(updated[feature])

        if feature.lower() == "ph":
            value = value - 0.35 * span
            value = max(info["min"], value)

        elif feature.lower() in {
            "turbidity",
            "solids",
            "chloramines",
            "trihalomethanes",
            "organic_carbon",
        }:
            value = value + 0.55 * span
            value = min(info["max"], value)

        elif feature.lower() in {"conductivity", "hardness", "sulfate"}:
            value = value + 0.30 * span
            value = min(info["max"], value)

        else:
            value = value + 0.20 * span
            value = min(info["max"], value)

        updated[feature] = float(np.clip(value, info["min"], info["max"]))

    return updated


def stream_step(sample):
    """
    Small data-derived random movement used only for demonstration of
    changing sensor readings. Every new reading is sent to the SVM.
    """
    updated = dict(sample)
    rng = np.random.default_rng()

    for feature in FEATURES:
        info = RANGES.get(feature, {})

        if "min" not in info:
            continue

        scale = max((info["p95"] - info["p05"]) * 0.035, 1e-9)
        value = float(updated[feature]) + rng.normal(0, scale)

        updated[feature] = float(
            np.clip(value, info["min"], info["max"])
        )

    return updated



def sample_changed(a, b, tolerance=1e-9):
    if a is None or b is None:
        return True

    for feature in FEATURES:
        try:
            if abs(float(a[feature]) - float(b[feature])) > tolerance:
                return True
        except (TypeError, ValueError):
            if a[feature] != b[feature]:
                return True

    return False


def record_history(sample, prediction, probability, source="Interactive"):
    if probability is None:
        return

    history = st.session_state.monitor_history

    if history and not sample_changed(sample, history[-1]["sample"]):
        return

    history.append(
        {
            "time": time.strftime("%H:%M:%S"),
            "prediction": prediction_label(prediction),
            "probability": float(probability),
            "sample": dict(sample),
            "source": source,
        }
    )

    # Keep the latest 30 readings so the page stays responsive.
    st.session_state.monitor_history = history[-30:]


def render_history():
    history = st.session_state.monitor_history

    if len(history) < 2:
        st.caption("Move a sensor or start the automatic stream to build prediction history.")
        return

    history_df = pd.DataFrame(
        {
            "Time": [row["time"] for row in history],
            "Potable Probability": [row["probability"] for row in history],
            "Prediction": [row["prediction"] for row in history],
            "Mode": [row["source"] for row in history],
        }
    )

    st.markdown(
        '<div class="history-title">Prediction history</div>',
        unsafe_allow_html=True,
    )

    chart_df = history_df[["Potable Probability"]].copy()
    chart_df.index = history_df["Time"]

    st.line_chart(
        chart_df,
        height=240,
        y="Potable Probability",
    )

    display_df = history_df.copy()
    display_df["Potable Probability"] = display_df["Potable Probability"].map(
        lambda x: f"{x:.1%}"
    )

    st.dataframe(
        display_df.iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )



def render_reading_grid(sample):
    columns = st.columns(3)

    for index, feature in enumerate(FEATURES):
        info = RANGES.get(feature, {})
        value = sample[feature]

        if isinstance(value, (float, int, np.floating, np.integer)):
            display_value = f"{float(value):.3f}"
        else:
            display_value = str(value)

        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="reading-box">
                    <div class="reading-name">{feature}</div>
                    <div class="reading-value">{display_value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_live_result(sample, mode_text, record=True, show_history=True):
    prediction, probability = predict_sample(sample)

    if record:
        record_history(
            sample,
            prediction,
            probability,
            source=(
                "Interactive"
                if mode_text == "INTERACTIVE SENSOR MODE"
                else "Automatic"
            ),
        )

    st.markdown(
        f'<span class="small-status">{mode_text}</span>',
        unsafe_allow_html=True,
    )

    prediction_card(
        prediction,
        probability,
        title="CURRENT MODEL PREDICTION",
    )

    a, b, c = st.columns(3)

    with a:
        st.metric("Prediction", prediction_label(prediction))

    with b:
        st.metric("Potable probability", pct(probability))

    with c:
        if probability is not None:
            st.metric("Non-Potable probability", f"{1 - probability:.1%}")
        else:
            st.metric("Input parameters", len(FEATURES))

    st.markdown(
        '<div class="section-title">Current readings</div>',
        unsafe_allow_html=True,
    )
    render_reading_grid(sample)

    if show_history:
        st.markdown("")
        render_history()

    return prediction, probability


# -----------------------------
# Session state
# -----------------------------
defaults = {
    "page": "Home",
    "analysis_result": None,
    "live_mode": "Interactive Sensors",
    "live_sample": make_baseline(),
    "live_before": None,
    "monitoring": False,
    "last_tick": time.time(),
    "monitor_history": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# Sidebar navigation
# -----------------------------
with st.sidebar:
    st.markdown("## AquaGuard")
    st.caption("Water-quality classification system")

    pages = [
        "Home",
        "Water Analysis",
        "Live Monitoring",
        "Model Performance",
        "EDA & Insights",
        "Methodology",
        "Model Information",
    ]

    page = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.page),
    )
    st.session_state.page = page

    st.divider()

    st.markdown("**Project type**")
    st.caption("Binary classification")

    st.markdown("**Selected model**")
    st.caption(SELECTED)

    st.divider()

    st.markdown(
        """
        <div class="sidebar-note">
        This is an academic machine-learning demonstration.
        A model prediction should not be treated as laboratory
        certification or a replacement for water-quality testing.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HOME
# ============================================================
if page == "Home":
    st.markdown('<div class="app-title">AquaGuard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        "Real-Time Water Safety Monitoring & Classification System"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-panel">'
        '<div class="hero-kicker">Machine Learning Classification Project</div>'
        '<div class="hero-heading">'
        "From water-quality measurements to an interactive model prediction"
        "</div>"
        '<div class="hero-copy">'
        "AquaGuard uses nine water-quality parameters and a trained Support "
        "Vector Machine to classify a sample as Potable or Non-Potable. "
        "The application combines exploratory analysis, model evaluation, "
        "single-sample prediction, and simulated real-time monitoring in one interface."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Project objectives</div>',
        unsafe_allow_html=True,
    )

    objective_cols = st.columns(3)
    objectives = [
        (
            "01",
            "Water-quality classification",
            "Build a binary classification system that learns patterns from "
            "the supplied water-potability dataset and produces a Potable / "
            "Non-Potable model prediction.",
        ),
        (
            "02",
            "Evaluation & model selection",
            "Compare multiple classification algorithms using cross-validation "
            "and select the final model using a predefined evaluation metric.",
        ),
        (
            "03",
            "Real-Time Application & Innovation",
            "Provide an interactive monitoring interface for future usability "
            "and innovation, where changing sensor readings are continuously "
            "re-evaluated by the trained model.",
        ),
    ]

    for col, (number, title, copy) in zip(objective_cols, objectives):
        with col:
            st.markdown(
                f'<div class="objective-card">'
                f'<div class="objective-number">{number}</div>'
                f'<div class="objective-title">{title}</div>'
                f'<div class="objective-copy">{copy}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Project procedure</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mini-flow">'
        '<div class="mini-flow-item"><div class="mini-flow-num">01</div>'
        '<div class="mini-flow-title">Understand</div></div>'
        '<div class="mini-flow-arrow">→</div>'
        '<div class="mini-flow-item"><div class="mini-flow-num">02</div>'
        '<div class="mini-flow-title">Prepare & Explore</div></div>'
        '<div class="mini-flow-arrow">→</div>'
        '<div class="mini-flow-item"><div class="mini-flow-num">03</div>'
        '<div class="mini-flow-title">Train & Compare</div></div>'
        '<div class="mini-flow-arrow">→</div>'
        '<div class="mini-flow-item"><div class="mini-flow-num">04</div>'
        '<div class="mini-flow-title">Evaluate</div></div>'
        '<div class="mini-flow-arrow">→</div>'
        '<div class="mini-flow-item"><div class="mini-flow-num">05</div>'
        '<div class="mini-flow-title">Deploy & Monitor</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    ds = meta["dataset"]
    test_metrics = meta["model_metrics"]["held_out_test"]

    st.markdown(
        '<div class="section-title">Project at a glance</div>',
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    with a:
        st.metric("Dataset samples", f"{ds['cleaned_shape'][0]:,}")

    with b:
        st.metric("Input parameters", len(FEATURES))

    with c:
        st.markdown(
            f"""
            <div class="objective-card" style="min-height:0; padding:0.72rem 0.9rem;">
                <div class="reading-name">Selected model</div>
                <div style="font-size:1.28rem; font-weight:720; color:var(--ag-navy);
                            line-height:1.25; white-space:normal;">
                    {SELECTED}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d:
        st.metric("Held-out F1", f"{test_metrics['F1']:.3f}")

    st.markdown(
        '<div class="callout">'
        '<strong>Dataset source:</strong> '
        '<a href="https://www.kaggle.com/datasets/adityakadiwal/water-potability/data" '
        'target="_blank" style="color:#0F7F80; font-weight:700; text-decoration:none;">'
        'Kaggle — Water Potability Dataset</a>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">What happens inside AquaGuard?</div>',
        unsafe_allow_html=True,
    )

    procedure_cols = st.columns(5)
    procedure = [
        ("01", "Input", "Nine water-quality measurements are provided."),
        ("02", "Preprocess", "Missing values are handled and numeric inputs are standardized."),
        ("03", "Classify", "The persisted SVM receives the processed feature vector."),
        ("04", "Evaluate", "The prediction and potable probability are returned."),
        ("05", "Monitor", "Interactive or simulated readings can be evaluated repeatedly."),
    ]

    for col, (number, title, copy) in zip(procedure_cols, procedure):
        with col:
            st.markdown(
                f'<div class="procedure-card">'
                f'<div class="procedure-step">Step {number}</div>'
                f'<div class="procedure-title">{title}</div>'
                f'<div class="procedure-copy">{copy}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Choose a demonstration</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Water Analysis")
        st.write(
            "Enter one new water-quality sample and send it through the "
            "saved preprocessing + SVM pipeline."
        )
        if st.button("Open Water Analysis", use_container_width=True):
            st.session_state.page = "Water Analysis"
            st.rerun()

    with right:
        st.markdown("### Live Monitoring")
        st.write(
            "Adjust simulated sensors yourself or run an automatic stream "
            "and observe how the model responds to changing readings."
        )
        if st.button("Open Live Monitoring", use_container_width=True):
            st.session_state.page = "Live Monitoring"
            st.rerun()

    st.markdown(
        '<div class="callout">'
        "<strong>Project scope:</strong> AquaGuard is an academic machine-learning "
        "demonstration. Its output is a model prediction learned from the supplied "
        "dataset; it is not laboratory certification or a replacement for "
        "water-quality testing or regulatory assessment."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# WATER ANALYSIS
# ============================================================
elif page == "Water Analysis":
    st.markdown('<div class="app-title">Water Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        "Single-sample prediction using the persisted preprocessing + SVM pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="info-strip">Enter a new sample below. The values are '
                'not used to retrain the model; they are only passed through '
                'the saved pipeline for prediction.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">Water-quality measurements</div>', unsafe_allow_html=True)

    values = {}
    columns = st.columns(3)

    for index, feature in enumerate(FEATURES):
        info = RANGES.get(feature, {})
        col = columns[index % 3]

        with col:
            if "min" in info:
                minimum = float(info["min"])
                maximum = float(info["max"])
                median = float(info["median"])

                step = max((maximum - minimum) / 1000, 0.01)

                values[feature] = st.number_input(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=median,
                    step=step,
                    format="%.4f",
                    key=f"analysis_{feature}",
                )

                st.caption(
                    f"Observed range: {minimum:.3f} – {maximum:.3f}"
                )
            else:
                categories = info.get("categories", [])
                values[feature] = st.selectbox(
                    feature,
                    categories,
                    key=f"analysis_{feature}",
                )

    st.markdown("")

    if st.button(
        "Analyze Water Sample",
        type="primary",
        use_container_width=True,
    ):
        prediction, probability = predict_sample(values)
        st.session_state.analysis_result = (
            dict(values),
            prediction,
            probability,
        )

    result = st.session_state.analysis_result

    if result is not None:
        sample, prediction, probability = result

        st.divider()
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

        prediction_card(
            prediction,
            probability,
            title="MODEL PREDICTION",
        )

        outside = outside_training_range(sample)

        if not outside.empty:
            st.warning(
                "One or more inputs are outside the observed training "
                "range. This is a distribution warning only; it does not "
                "by itself mean the water is unsafe."
            )
            st.dataframe(
                outside,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown('<div class="section-title">Input summary</div>', unsafe_allow_html=True)
        st.dataframe(
            feature_table(sample),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# LIVE MONITORING
# ============================================================
elif page == "Live Monitoring":
    st.markdown('<div class="app-title">Live Monitoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        "Interact with changing sensor readings and watch the persisted SVM "
        "reclassify every new state."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-strip">'
        "<strong>Important:</strong> this is a simulated monitoring stream. "
        "It does not connect to physical sensors. The application changes "
        "input values and sends those values to the trained SVM."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Monitoring mode</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Choose how readings should change",
        ["Interactive Sensors", "Automatic Stream"],
        horizontal=True,
        index=0 if st.session_state.live_mode == "Interactive Sensors" else 1,
        key="live_mode_radio",
    )
    st.session_state.live_mode = mode

    # --------------------------------------------------------
    # Interactive mode
    # --------------------------------------------------------
    if mode == "Interactive Sensors":
        st.markdown("### Interactive sensor controls")
        st.markdown(
            '<div class="mode-note">'
            "Adjust the sensor values below. Every change creates a new "
            "nine-feature input vector and is immediately passed through "
            "the persisted SVM. The prediction and potable probability "
            "update from the model output."
            "</div>",
            unsafe_allow_html=True,
        )

        interactive_sample = {}

        columns = st.columns(3)

        for index, feature in enumerate(FEATURES):
            info = RANGES.get(feature, {})
            col = columns[index % 3]

            with col:
                minimum = float(info["min"])
                maximum = float(info["max"])
                current = float(
                    st.session_state.live_sample.get(
                        feature,
                        info["median"],
                    )
                )

                value = st.slider(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=float(np.clip(current, minimum, maximum)),
                    step=max((maximum - minimum) / 200, 0.01),
                    format="%.3f",
                    key=f"sensor_{feature}",
                )

                interactive_sample[feature] = value

        st.session_state.live_sample = dict(interactive_sample)

        st.markdown('<div class="section-title">Live model response</div>', unsafe_allow_html=True)

        render_live_result(
            st.session_state.live_sample,
            "INTERACTIVE SENSOR MODE",
        )

        if st.button("Reset Sensors to Baseline", use_container_width=True):
            st.session_state.live_sample = make_baseline()
            st.session_state.monitor_history = []
            st.session_state.live_before = None

            for feature in FEATURES:
                key = f"sensor_{feature}"
                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

    # --------------------------------------------------------
    # Automatic mode
    # --------------------------------------------------------
    else:
        st.markdown("### Automatic sensor stream")
        st.write(
            "The application generates a small change in each reading at "
            "regular intervals. Each generated reading is sent through the "
            "same saved SVM model."
        )

        control1, control2, control3, control4 = st.columns(4)

        with control1:
            if st.button("Start", type="primary", use_container_width=True):
                st.session_state.monitoring = True
                st.session_state.last_tick = time.time()

        with control2:
            if st.button("Pause", use_container_width=True):
                st.session_state.monitoring = False

        with control3:
            if st.button("Simulate Contamination", use_container_width=True):
                st.session_state.live_before = dict(
                    st.session_state.live_sample
                )
                st.session_state.live_sample = move_toward_contamination(
                    st.session_state.live_sample
                )
                st.session_state.monitoring = False

        with control4:
            if st.button("Restore Baseline", use_container_width=True):
                st.session_state.live_sample = make_baseline()
                st.session_state.live_before = None
                st.session_state.monitoring = False
                st.session_state.monitor_history = []

        if st.button("Reset Monitoring", use_container_width=True):
            st.session_state.live_sample = make_baseline()
            st.session_state.live_before = None
            st.session_state.monitoring = False
            st.session_state.monitor_history = []
            st.rerun()

        status_text = "RUNNING" if st.session_state.monitoring else "PAUSED"

        a, b, c = st.columns(3)

        with a:
            st.metric("Monitoring status", status_text)

        with b:
            st.metric("Update interval", "2 seconds")

        with c:
            st.metric("Model", SELECTED)

        def automatic_panel():
            if st.session_state.monitoring:
                now = time.time()

                if now - st.session_state.last_tick >= 1.8:
                    st.session_state.live_sample = stream_step(
                        st.session_state.live_sample
                    )
                    st.session_state.last_tick = now

            render_live_result(
                st.session_state.live_sample,
                "AUTOMATIC STREAM",
            )

            if st.session_state.monitoring:
                st.caption(
                    f"Last generated reading: "
                    f"{time.strftime('%H:%M:%S', time.localtime(st.session_state.last_tick))}"
                )

        if hasattr(st, "fragment"):
            if st.session_state.monitoring:
                @st.fragment(run_every="2s")
                def live_fragment():
                    automatic_panel()

                live_fragment()
            else:
                automatic_panel()
        else:
            automatic_panel()

            if st.session_state.monitoring:
                st.info(
                    "Automatic refresh is unavailable in this Streamlit "
                    "version. Click 'Advance Reading' to generate the next "
                    "reading."
                )
                if st.button("Advance Reading", type="primary"):
                    st.session_state.live_sample = stream_step(
                        st.session_state.live_sample
                    )
                    st.rerun()

        # Before/after comparison
        if st.session_state.live_before is not None:
            before = st.session_state.live_before
            after = st.session_state.live_sample

            before_pred, before_prob = predict_sample(before)
            after_pred, after_prob = predict_sample(after)

            st.divider()
            st.markdown(
                '<div class="section-title">Before / after comparison</div>',
                unsafe_allow_html=True,
            )

            left, right = st.columns(2)

            with left:
                st.markdown("#### Before")
                prediction_card(
                    before_pred,
                    before_prob,
                    title="PREVIOUS MODEL PREDICTION",
                )
                st.dataframe(
                    feature_table(before),
                    use_container_width=True,
                    hide_index=True,
                )

            with right:
                st.markdown("#### After")
                prediction_card(
                    after_pred,
                    after_prob,
                    title="CURRENT MODEL PREDICTION",
                )
                st.dataframe(
                    feature_table(after),
                    use_container_width=True,
                    hide_index=True,
                )

            changed_rows = []

            for feature in FEATURES:
                before_value = float(before[feature])
                after_value = float(after[feature])
                delta = after_value - before_value

                if abs(delta) > 1e-9:
                    changed_rows.append(
                        {
                            "Parameter": feature,
                            "Before": before_value,
                            "After": after_value,
                            "Change": delta,
                        }
                    )

            if changed_rows:
                st.markdown(
                    '<div class="section-title">Changed parameters</div>',
                    unsafe_allow_html=True,
                )

                changed_df = pd.DataFrame(changed_rows)
                st.dataframe(
                    changed_df.style.format(
                        {
                            "Before": "{:.3f}",
                            "After": "{:.3f}",
                            "Change": "{:+.3f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            if before_pred != after_pred:
                st.success(
                    "The SVM classification changed after the simulated "
                    "measurement change."
                )
            else:
                st.info(
                    "The readings changed, but the SVM classification "
                    "remained the same."
                )

        st.markdown(
            '<div class="muted">'
            "The contamination control changes measurements using the "
            "observed training-data distribution. It does not force a "
            "particular prediction."
            "</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.markdown('<div class="app-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        "Comparison of the classification models and final held-out test evaluation."
        "</div>",
        unsafe_allow_html=True,
    )

    comparison_path = ROOT / "outputs" / "model_comparison.csv"

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)

        st.markdown(
            f"**Selected model:** {SELECTED}  \n"
            f"**Selection metric:** {meta['selection_metric']} — highest "
            "5-fold cross-validated F1 on the training split."
        )

        display_columns = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "NonPotable_Recall",
        ]

        display = comparison[display_columns].copy()

        st.dataframe(
            display.style.format(
                {column: "{:.3f}" for column in display_columns[1:]}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown('<div class="section-title">Held-out test set</div>', unsafe_allow_html=True)

        metrics = meta["model_metrics"]["held_out_test"]

        metric_columns = st.columns(5)

        for column, (name, value) in zip(
            metric_columns,
            [
                ("Accuracy", metrics["Accuracy"]),
                ("Precision", metrics["Precision"]),
                ("Recall", metrics["Recall"]),
                ("F1", metrics["F1"]),
                ("ROC-AUC", metrics["ROC_AUC"]),
            ],
        ):
            with column:
                st.metric(name, f"{value:.3f}")

        image_left, image_right = st.columns(2)

        with image_left:
            confusion_path = EVAL_DIR / "confusion_matrix.png"
            if confusion_path.exists():
                st.image(
                    str(confusion_path),
                    caption="Held-out test confusion matrix",
                    use_container_width=True,
                )

        with image_right:
            roc_path = EVAL_DIR / "roc_curve.png"
            if roc_path.exists():
                st.image(
                    str(roc_path),
                    caption="Held-out test ROC curve",
                    use_container_width=True,
                )

        comparison_plot = EDA_DIR / "12_model_comparison.png"
        if comparison_plot.exists():
            st.image(
                str(comparison_plot),
                caption="5-fold cross-validated model comparison",
                use_container_width=True,
            )

        st.caption(
            f"Non-Potable recall on the held-out test set: "
            f"{metrics['NonPotable_Recall']:.3f}."
        )

    else:
        st.warning(
            "Model comparison results were not found. Run "
            "`python src\\train_models.py` first."
        )


# ============================================================
# EDA
# ============================================================
elif page == "EDA & Insights":
    st.markdown('<div class="app-title">EDA & Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        "Exploratory analysis of the supplied water-potability dataset."
        "</div>",
        unsafe_allow_html=True,
    )

    sections = [
        (
            "Class distribution",
            "01_class_distribution.png",
            "The target contains two classes: Non-Potable and Potable. "
            "The class counts are not equal, so the dataset has moderate "
            "class imbalance.",
        ),
        (
            "Missing values",
            "02_missing_values.png",
            "Missing measurements are concentrated in pH, Sulfate, and "
            "Trihalomethanes. Numeric missing values are handled by median "
            "imputation inside the training pipeline.",
        ),
        (
            "Feature distributions",
            "03_feature_distributions.png",
            "The nine measurements have different scales and distributions. "
            "Standardization is therefore useful for scale-sensitive models.",
        ),
        (
            "Outliers",
            "04_feature_boxplots.png",
            "Boxplots identify extreme observations. They were retained "
            "rather than deleting every statistical outlier.",
        ),
        (
            "Correlation",
            "05_correlation_heatmap.png",
            "The heatmap shows pairwise linear relationships. Correlation "
            "does not establish causation.",
        ),
        (
            "Features by class",
            "06_features_by_class.png",
            "Feature distributions are compared across the two observed "
            "target classes. Considerable overlap remains.",
        ),
        (
            "Key relationships",
            "07_key_feature_relationships.png",
            "Selected feature-versus-target relationships provide additional "
            "descriptive views of the dataset.",
        ),
        (
            "Pairwise relationships",
            "08_pairwise_relationships.png",
            "Selected feature pairs are inspected for joint patterns.",
        ),
        (
            "PCA — explained variance",
            "09_pca_explained_variance.png",
            "Shows how much standardized numeric variance is represented by "
            "each principal component.",
        ),
        (
            "PCA — cumulative variance",
            "10_pca_cumulative_variance.png",
            "Shows how many principal components are needed to retain most "
            "of the standardized numeric variance.",
        ),
        (
            "PCA — 2D projection",
            "11_pca_2d.png",
            "A compact two-dimensional view of the training data. Overlap "
            "between classes indicates that two principal dimensions do "
            "not perfectly separate the classes.",
        ),
    ]

    for title, filename, description in sections:
        image_path = EDA_DIR / filename

        if image_path.exists():
            with st.expander(title, expanded=(title == "Class distribution")):
                st.image(str(image_path), use_container_width=True)
                st.caption(description)

    pca = meta.get("pca_analysis", {})

    if pca:
        st.markdown('<div class="section-title">PCA summary</div>', unsafe_allow_html=True)
        st.write(
            f"{pca['components_for_80_percent']} components explain at least "
            f"80% of standardized numeric variance, while "
            f"{pca['components_for_90_percent']} components explain at least "
            f"90%."
        )
        st.caption(
            "PCA was used as an analytical dimensionality-reduction step. "
            "It was not forced into the final production classifier."
        )


# ============================================================
# METHODOLOGY
# ============================================================
elif page == "Methodology":
    st.markdown('<div class="app-title">Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        "The complete procedure followed to develop, evaluate, and integrate AquaGuard."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-panel">'
        '<div class="hero-kicker">End-to-End ML Procedure</div>'
        '<div class="hero-heading">'
        "A structured pipeline from dataset understanding to real-time interaction"
        "</div>"
        '<div class="hero-copy">'
        "The project follows a sequence of data understanding, preparation, "
        "exploration, model development, evaluation, persistence, and application "
        "integration. The held-out test set is kept separate from model selection."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Four project phases</div>',
        unsafe_allow_html=True,
    )

    phase_cols = st.columns(4)

    phases = [
        (
            "PHASE 01",
            "Understand the data",
            "Define the classification problem and establish what the dataset contains.",
            [
                "Problem definition",
                "Dataset dimensions and data types",
                "Target distribution",
                "Missing values and duplicates",
            ],
        ),
        (
            "PHASE 02",
            "Prepare & explore",
            "Clean the inputs and investigate distributions, relationships, outliers, and dimensionality.",
            [
                "Duplicate check",
                "Median imputation",
                "Feature standardization",
                "EDA + PCA analysis",
            ],
        ),
        (
            "PHASE 03",
            "Train & evaluate",
            "Compare candidate classifiers and select the final approach using a predefined metric.",
            [
                "80/20 stratified split",
                "5-fold cross-validation",
                "Six classification models",
                "F1-based model selection",
            ],
        ),
        (
            "PHASE 04",
            "Deploy & monitor",
            "Persist the selected pipeline and connect it to an interactive Streamlit application.",
            [
                "Save preprocessing + SVM",
                "Unseen sample prediction",
                "Interactive sensor mode",
                "Automatic simulated monitoring",
            ],
        ),
    ]

    for col, (tag, title, copy, items) in zip(phase_cols, phases):
        with col:
            bullets = "".join(f"<li>{item}</li>" for item in items)
            st.markdown(
                f'<div class="phase-card">'
                f'<div class="phase-tag">{tag}</div>'
                f'<div class="phase-title">{title}</div>'
                f'<div class="phase-copy">{copy}</div>'
                f'<ul class="phase-list">{bullets}</ul>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Detailed procedure</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "1. Data & EDA",
            "2. Preprocessing & PCA",
            "3. Model Development",
            "4. Evaluation & Selection",
            "5. Application",
        ]
    )

    with tabs[0]:
        st.markdown("#### Problem and dataset")
        st.write(
            "AquaGuard treats water potability as a binary classification problem. "
            "The target is Potability, with 0 representing Non-Potable and 1 representing Potable."
        )

        st.markdown("#### Exploratory analysis")
        st.write(
            "The dataset was inspected for dimensions, data types, missing values, "
            "duplicates, descriptive statistics, class distribution, feature distributions, "
            "outliers, correlations, class-wise patterns, and pairwise relationships."
        )

        st.markdown("#### Why class distribution matters")
        st.write(
            "The target classes are not equally represented. Therefore, model evaluation "
            "uses precision, recall, and F1 in addition to accuracy rather than relying on accuracy alone."
        )

    with tabs[1]:
        st.markdown("#### Data preparation")
        st.write(
            "Exact duplicates were checked. Numeric missing values are handled using median "
            "imputation inside the persisted preprocessing pipeline, so the same transformation "
            "is applied during both training and prediction."
        )

        st.markdown("#### Standardization")
        st.write(
            "Numeric features are standardized using StandardScaler. This is particularly relevant "
            "for scale-sensitive models such as SVM and KNN."
        )

        st.markdown("#### PCA")
        pca = meta.get("pca_analysis", {})
        if pca:
            st.write(
                f"{pca['components_for_80_percent']} principal components explain at least "
                f"80% of standardized numeric variance, while "
                f"{pca['components_for_90_percent']} explain at least 90%."
            )
        st.write(
            "PCA is used as an analytical dimensionality-reduction step to understand the "
            "structure and explained variance of the feature space. It is not forced into "
            "the final production prediction pipeline."
        )

    with tabs[2]:
        st.markdown("#### Train/test strategy")
        st.write(
            "The cleaned labelled dataset is divided using an 80/20 stratified split. "
            "The training portion is used for cross-validation and model selection, while "
            "the held-out test set is reserved for final evaluation."
        )

        st.markdown("#### Models compared")
        st.write(
            "Logistic Regression, KNN, Decision Tree, Random Forest, Support Vector Machine (SVM), "
            "and XGBoost were compared using the same preprocessing approach."
        )

        st.markdown("#### Model pipeline")
        st.write(
            "Each candidate model is combined with the preprocessing pipeline. This keeps "
            "preprocessing and classification together and reduces the risk of applying "
            "different transformations during deployment."
        )

    with tabs[3]:
        st.markdown("#### Cross-validation")
        st.write(
            "Five-fold StratifiedKFold cross-validation is performed on the training split. "
            "Out-of-fold predictions are used to calculate the comparison metrics."
        )

        st.markdown("#### Selection rule")
        st.write(
            f"The predefined selection metric is **{meta['selection_metric']}**. "
            f"The selected classifier is **{SELECTED}**, based on the highest cross-validated "
            "F1 score on the training split."
        )

        st.markdown("#### Final held-out evaluation")
        metrics = meta["model_metrics"]["held_out_test"]
        st.dataframe(
            pd.DataFrame(
                [
                    ["Accuracy", metrics["Accuracy"]],
                    ["Precision", metrics["Precision"]],
                    ["Recall", metrics["Recall"]],
                    ["F1", metrics["F1"]],
                    ["ROC-AUC", metrics["ROC_AUC"]],
                    ["Non-Potable Recall", metrics["NonPotable_Recall"]],
                ],
                columns=["Metric", "Value"],
            ).style.format({"Value": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[4]:
        st.markdown("#### Model persistence")
        st.write(
            "The complete preprocessing + selected classifier pipeline is stored in "
            "`models/best_model.pkl`. The Streamlit application loads this artifact "
            "without retraining."
        )

        st.markdown("#### Single-sample prediction")
        st.write(
            "A new set of nine measurements is converted into the same feature structure "
            "used during training and passed through the saved pipeline."
        )

        st.markdown("#### Real-time demonstration")
        st.write(
            "Interactive sensor sliders and an automatic simulated stream generate changing "
            "input states. Each state is evaluated by the same persisted SVM. The application "
            "records prediction probability history and supports before/after comparison."
        )

    st.markdown(
        '<div class="section-title">Why this procedure?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="callout">'
        "<strong>Key principle:</strong> the application separates analysis, model selection, "
        "final evaluation, and deployment. PCA helps understand the feature space, cross-validation "
        "supports model comparison, the held-out test set provides an unseen final check, and the "
        "persisted pipeline ensures the application uses the same preprocessing and classifier "
        "after training."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL INFORMATION
# ============================================================
elif page == "Model Information":
    st.markdown('<div class="app-title">Model Information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        "Dataset, model, preprocessing, PCA and evaluation details."
        "</div>",
        unsafe_allow_html=True,
    )

    dataset = meta["dataset"]
    metrics = meta["model_metrics"]["held_out_test"]

    a, b, c = st.columns(3)

    with a:
        st.metric("Problem", "Binary Classification")

    with b:
        st.metric("Target", TARGET)

    with c:
        st.metric("Selected model", SELECTED)

    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-strip">'
        '<strong>Source:</strong> '
        '<a href="https://www.kaggle.com/datasets/adityakadiwal/water-potability/data" '
        'target="_blank" style="color:#0F7F80; font-weight:700;">'
        'Kaggle — Water Potability Dataset</a>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"Original dataset: **{dataset['original_shape'][0]:,} rows × "
        f"{dataset['original_shape'][1]} columns**."
    )

    st.write(
        f"Training set: **{dataset['training_shape'][0]:,} samples** | "
        f"Test set: **{dataset['test_shape'][0]:,} samples**."
    )

    st.write(
        f"Target mapping: **0 = Non-Potable**, **1 = Potable**."
    )

    st.markdown('<div class="section-title">Input features</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({"Feature": FEATURES}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-title">Preprocessing</div>', unsafe_allow_html=True)

    for step in meta.get("preprocessing_steps", []):
        st.write("•", step)

    st.markdown('<div class="section-title">PCA</div>', unsafe_allow_html=True)

    pca = meta.get("pca_analysis", {})

    if pca:
        st.write(
            f"{pca['components_for_80_percent']} components reach at least "
            f"80% cumulative explained variance and "
            f"{pca['components_for_90_percent']} reach at least 90%."
        )

    st.caption(
        "PCA is an analytical component of the project and is not included "
        "in the final production prediction pipeline."
    )

    st.markdown('<div class="section-title">Held-out test results</div>', unsafe_allow_html=True)

    results = pd.DataFrame(
        [
            {
                "Metric": "Accuracy",
                "Value": metrics["Accuracy"],
            },
            {
                "Metric": "Precision",
                "Value": metrics["Precision"],
            },
            {
                "Metric": "Recall",
                "Value": metrics["Recall"],
            },
            {
                "Metric": "F1",
                "Value": metrics["F1"],
            },
            {
                "Metric": "ROC-AUC",
                "Value": metrics["ROC_AUC"],
            },
            {
                "Metric": "Non-Potable Recall",
                "Value": metrics["NonPotable_Recall"],
            },
        ]
    )

    results["Value"] = results["Value"].map(lambda value: f"{value:.4f}")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-title">Prediction limitation</div>', unsafe_allow_html=True)

    st.warning(
        "AquaGuard produces a machine-learning prediction from the supplied "
        "dataset. It is not laboratory certification, medical advice, "
        "regulatory compliance, or a guarantee that water is safe to drink."
    )
