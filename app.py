import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import html as html_lib
from PIL import Image
from datetime import date
from scipy.optimize import milp, LinearConstraint, Bounds


# =========================================================
# PAGE CONFIG
# =========================================================

transparent_icon = Image.new("RGBA", (32, 32), (0, 0, 0, 0))

st.set_page_config(
    page_title="Student Early Warning System",
    page_icon=transparent_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# DESIGN / CSS
# =========================================================

st.markdown(
    """
    <style>
        /* Hide Streamlit UI */
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stHeaderActionElements"],
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
        }

        header {
            visibility: hidden !important;
            height: 0 !important;
        }

        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
        [data-testid="stHeadingWithActionElements"] a {
            display: none !important;
        }

        /* Global */
        .stApp {
            background: #F7F9FC;
            color: #0F1F35;
        }

        .block-container {
            max-width: 1600px;
            padding-top: 1.0rem;
            padding-bottom: 2rem;
            padding-left: 1.15rem;
            padding-right: 1.15rem;
        }

        h1, h2, h3, h4 {
            color: #0E2038 !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #0E223C;
            border-right: 1px solid #163554;
            min-width: 245px;
            max-width: 245px;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
        }

        [data-testid="stSidebar"] * {
            color: #FFFFFF;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 8px;
            padding: 0.58rem 0.72rem;
            margin-bottom: 0.25rem;
            transition: 0.15s ease-in-out;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: #173A61;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: #1E73E8;
        }

        [data-testid="stSidebar"] .stRadio > label {
            display: none;
        }

        /* Top bar */
        .topbar {
            background: #FFFFFF;
            border: 1px solid #E7ECF3;
            border-radius: 10px;
            padding: 0.7rem 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 1px 3px rgba(15, 31, 53, 0.04);
            margin-bottom: 1.1rem;
        }

        .search-box {
            background: #F3F7FC;
            border: 1px solid #E7EDF5;
            color: #7B899B;
            border-radius: 7px;
            padding: 0.48rem 0.8rem;
            width: 330px;
            font-size: 0.82rem;
        }

        .top-date {
            color: #53657A;
            font-size: 0.82rem;
            font-weight: 600;
        }

        /* Welcome */
        .welcome {
            margin-bottom: 1rem;
        }

        .welcome-title {
            color: #0D1F36;
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .welcome-sub {
            color: #6B7A90;
            font-size: 0.9rem;
            margin-top: 0.18rem;
        }

        /* Summary cards */
        .summary-card {
            border-radius: 10px;
            padding: 1rem 1.05rem;
            min-height: 132px;
            border: 1px solid;
            box-shadow: 0 2px 8px rgba(15,31,53,0.035);
        }

        .card-blue {
            background: #F3F8FF;
            border-color: #D6E6FB;
        }

        .card-red {
            background: #FFF4F4;
            border-color: #F5D4D4;
        }

        .card-yellow {
            background: #FFF9E9;
            border-color: #F3E2AA;
        }

        .card-green {
            background: #F1FBF6;
            border-color: #CDEBDB;
        }

        .summary-row {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }

        .summary-icon {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            font-weight: 800;
        }

        .icon-blue { background:#DCEAFF; color:#246DD8; }
        .icon-red { background:#FFE0E0; color:#E63B3B; }
        .icon-yellow { background:#FFF0B8; color:#D99A00; }
        .icon-green { background:#D7F3E4; color:#16A66A; }

        .summary-label {
            color: #475A70;
            font-size: 0.86rem;
            font-weight: 700;
        }

        .summary-value {
            color: #0D1F36;
            font-size: 1.85rem;
            font-weight: 800;
            margin-top: 0.45rem;
        }

        .summary-sub {
            color: #6F7E91;
            font-size: 0.78rem;
            margin-top: 0.3rem;
        }

        .red-text { color:#E34242; }
        .yellow-text { color:#C88A00; }
        .green-text { color:#15975F; }
        .blue-text { color:#2471D8; }

        /* White panels */
        .panel-title {
            font-size: 0.98rem;
            font-weight: 800;
            color: #142840;
            margin-bottom: 0.15rem;
        }

        .panel-sub {
            font-size: 0.76rem;
            color: #7A899B;
            margin-bottom: 0.45rem;
        }

        .panel-shell {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.9rem;
            box-shadow: 0 1px 5px rgba(15,31,53,0.035);
        }

        .chart-card {
            background: #FFFFFF;
            border: 1px solid #E3E8EF;
            border-radius: 12px;
            padding: 0.95rem 1rem 0.65rem 1rem;
            box-shadow: 0 2px 8px rgba(15,31,53,0.035);
            min-height: 315px;
        }

        /* Student detail */
        .student-detail {
            background:#FFFFFF;
            border:1px solid #E2E8F0;
            border-radius:10px;
            padding:1rem;
            min-height:280px;
        }

        .student-id {
            font-weight:800;
            color:#10243D;
            font-size:1.05rem;
        }

        .risk-pill {
            display:inline-block;
            padding:0.24rem 0.55rem;
            border-radius:999px;
            font-size:0.72rem;
            font-weight:800;
        }

        .pill-low {background:#E7F7EF;color:#17885A;}
        .pill-medium {background:#FFF3BF;color:#9A7000;}
        .pill-high {background:#FFE1B8;color:#B96500;}
        .pill-very-high {background:#FFDCDC;color:#C83131;}

        .detail-label {
            color:#718095;
            font-size:0.72rem;
            margin-top:0.55rem;
        }

        .detail-value {
            color:#152A43;
            font-size:0.87rem;
            font-weight:650;
        }


        /* Interactive dashboard summary cards */
        .st-key-card_total button,
        .st-key-card_at_risk button,
        .st-key-card_very_high button,
        .st-key-card_on_track button {
            width: 100%;
            min-height: 154px;
            border-radius: 10px;
            padding: 1rem 1.15rem;
            text-align: left !important;
            justify-content: flex-start !important;
            white-space: pre-line !important;
            box-shadow: 0 2px 8px rgba(15,31,53,0.035);
            transition: transform 0.14s ease, box-shadow 0.14s ease;
        }

        .st-key-card_total button {
            background: #F3F8FF !important;
            border: 1px solid #D6E6FB !important;
        }

        .st-key-card_at_risk button {
            background: #FFF4F4 !important;
            border: 1px solid #F5D4D4 !important;
        }

        .st-key-card_very_high button {
            background: #FFF9E9 !important;
            border: 1px solid #F3E2AA !important;
        }

        .st-key-card_on_track button {
            background: #F1FBF6 !important;
            border: 1px solid #CDEBDB !important;
        }

        .st-key-card_total button:hover,
        .st-key-card_at_risk button:hover,
        .st-key-card_very_high button:hover,
        .st-key-card_on_track button:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 18px rgba(15,31,53,0.10);
        }

        .st-key-card_total button p,
        .st-key-card_at_risk button p,
        .st-key-card_very_high button p,
        .st-key-card_on_track button p {
            white-space: pre-line !important;
            text-align: left !important;
            color: #0D1F36 !important;
            line-height: 1.55 !important;
            font-weight: 700 !important;
        }

        [data-testid="stDataFrame"] {
            background: #FFFFFF !important;
            border: 1px solid #E3E8EF !important;
            border-radius: 9px !important;
            overflow: hidden !important;
        }

        [data-testid="stDataFrame"] > div {
            background: #FFFFFF !important;
        }

        .selected-segment-panel {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.4rem 0 1.1rem 0;
            box-shadow: 0 1px 5px rgba(15,31,53,0.035);
        }

        .selected-segment-title {
            color: #10243D;
            font-size: 1.08rem;
            font-weight: 800;
        }

        .selected-segment-sub {
            color: #728096;
            font-size: 0.8rem;
            margin-top: 0.18rem;
        }


        /* Stronger colour language on overview cards */
        .st-key-card_total button {
            border-left: 5px solid #2F80ED !important;
        }

        .st-key-card_at_risk button {
            border-left: 5px solid #E5484D !important;
            background: #FFF1F2 !important;
        }

        .st-key-card_very_high button {
            border-left: 5px solid #F2B824 !important;
            background: #FFF8E1 !important;
        }

        .st-key-card_on_track button {
            border-left: 5px solid #22A06B !important;
            background: #EEFAF4 !important;
        }

        /* Compact chart headings */
        .dashboard-chart-title {
            color: #142840;
            font-size: 0.93rem;
            font-weight: 800;
            margin-bottom: 0.05rem;
        }

        .dashboard-chart-sub {
            color: #7A899B;
            font-size: 0.72rem;
            margin-bottom: 0.3rem;
        }

        /* Selected overview colour states */
        .segment-blue {
            background: #F3F8FF;
            border-left: 5px solid #2F80ED;
        }

        .segment-red {
            background: #FFF1F2;
            border-left: 5px solid #E5484D;
        }

        .segment-yellow {
            background: #FFF8E1;
            border-left: 5px solid #F2B824;
        }

        .segment-green {
            background: #EEFAF4;
            border-left: 5px solid #22A06B;
        }

        /* Light dashboard tables */
        .ews-table-wrap {
            width: 100%;
            overflow-x: auto;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            margin-top: 0.35rem;
            margin-bottom: 0.7rem;
        }

        .ews-table {
            width: 100%;
            border-collapse: collapse;
            color: #24364B;
            font-size: 0.78rem;
        }

        .ews-table thead th {
            background: #F5F8FC;
            color: #56677C;
            font-size: 0.72rem;
            font-weight: 800;
            text-align: left;
            padding: 0.72rem 0.75rem;
            border-bottom: 1px solid #E2E8F0;
            white-space: nowrap;
        }

        .ews-table tbody td {
            background: #FFFFFF;
            padding: 0.72rem 0.75rem;
            border-bottom: 1px solid #EDF1F6;
            vertical-align: middle;
        }

        .ews-table tbody tr:last-child td {
            border-bottom: none;
        }

        .ews-table tbody tr:hover td {
            background: #F9FBFE;
        }

        .risk-badge {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            font-size: 0.69rem;
            font-weight: 800;
            white-space: nowrap;
        }

        .risk-low {
            background: #DDF5E9;
            color: #137A50;
        }

        .risk-medium {
            background: #FFF0B8;
            color: #906700;
        }

        .risk-high {
            background: #FFE2BF;
            color: #A85C00;
        }

        .risk-very-high {
            background: #FFDCDD;
            color: #BD3036;
        }

        .score-track {
            width: 92px;
            height: 7px;
            background: #EDF1F6;
            border-radius: 999px;
            overflow: hidden;
            display: inline-block;
            margin-right: 0.4rem;
            vertical-align: middle;
        }

        .score-fill {
            height: 100%;
            border-radius: 999px;
        }

        .student-detail-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.8rem 1rem;
            margin-top: 0.8rem;
        }

        .student-detail-item {
            background: #F8FAFD;
            border: 1px solid #EDF1F6;
            border-radius: 8px;
            padding: 0.7rem;
        }

        @media (max-width: 900px) {
            .student-detail-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Streamlit widgets */
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 7px;
            font-weight: 700;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #E3E8EF;
            border-radius: 9px;
            overflow: hidden;
        }

        div[data-testid="stSelectbox"] > div,
        div[data-testid="stTextInput"] > div {
            border-radius: 7px;
        }

        /* =========================================================
           COLOUR-CODED DASHBOARD SUMMARY BUTTONS
           Compatible with older Streamlit versions.
           Targets the dashboard's unique four-button row by position.
           ========================================================= */

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(1) button,
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(2) button,
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(3) button,
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(4) button {
            width: 100% !important;
            min-height: 92px !important;
            border-radius: 10px !important;
            padding: 0.5rem 0.7rem !important;
            white-space: pre-line !important;
            box-shadow: 0 2px 8px rgba(15,31,53,0.04) !important;
            transition: transform 0.14s ease, box-shadow 0.14s ease, background 0.14s ease !important;
        }

        /* 1. Total Students — BLUE */
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(1) button {
            background: #EEF5FF !important;
            border: 1.5px solid #2F80ED !important;
            border-left: 6px solid #2F80ED !important;
            color: #1F66C7 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(1) button p {
            color: #1F66C7 !important;
        }

        /* 2. At Risk Students — RED */
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(2) button {
            background: #FFF0F1 !important;
            border: 1.5px solid #E5484D !important;
            border-left: 6px solid #E5484D !important;
            color: #C9353A !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(2) button p {
            color: #C9353A !important;
        }

        /* 3. Very High Risk — YELLOW / GOLD */
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(3) button {
            background: #FFF7DE !important;
            border: 1.5px solid #F2B824 !important;
            border-left: 6px solid #F2B824 !important;
            color: #9A6B00 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(3) button p {
            color: #9A6B00 !important;
        }

        /* 4. On Track — GREEN */
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(4) button {
            background: #ECF9F2 !important;
            border: 1.5px solid #22A06B !important;
            border-left: 6px solid #22A06B !important;
            color: #168257 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(4) button p {
            color: #168257 !important;
        }

        /* Matching hover / focus colours so Streamlit blue doesn't take over */
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(1) button:hover,
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(1) button:focus,
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(1) button:active {
            background: #DCEBFF !important;
            border-color: #2F80ED !important;
            color: #1F66C7 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(2) button:hover,
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(2) button:focus,
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(2) button:active {
            background: #FFE0E2 !important;
            border-color: #E5484D !important;
            color: #C9353A !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(3) button:hover,
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(3) button:focus,
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(3) button:active {
            background: #FFEDB5 !important;
            border-color: #F2B824 !important;
            color: #9A6B00 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(4) button:hover,
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(4) button:focus,
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(4) button:active {
            background: #D9F2E5 !important;
            border-color: #22A06B !important;
            color: #168257 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(1) button:focus-visible {
            outline: 3px solid rgba(47,128,237,0.18) !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(2) button:focus-visible {
            outline: 3px solid rgba(229,72,77,0.18) !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(3) button:focus-visible {
            outline: 3px solid rgba(242,184,36,0.20) !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) > div:nth-child(4) button:focus-visible {
            outline: 3px solid rgba(34,160,107,0.18) !important;
        }


        /* Compact typography for dashboard summary buttons */
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) button p {
            line-height: 1.18 !important;
            font-size: 0.82rem !important;
            margin: 0 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4) button) button {
            letter-spacing: 0 !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FILES / CONSTANTS
# =========================================================

MODEL_FILE = "academic_risk_model.pkl"
DATA_FILE = "StudentPerformanceFactors.csv"
CV_RESULTS_FILE = "model_comparison_cross_validation.csv"
HELDOUT_METRICS_FILE = "logistic_regression_heldout_metrics.csv"
TARGET_SENSITIVITY_FILE = "target_threshold_sensitivity.csv"
WEIGHT_SENSITIVITY_FILE = "utility_weight_sensitivity.csv"

REQUIRED_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]

BASE_WEIGHTS = {
    "ml_risk": 0.60,
    "attendance_risk": 0.25,
    "previous_performance_risk": 0.15
}

RISK_ORDER = ["Low", "Medium", "High", "Very High"]

RISK_COLORS = {
    "Low": "#22A06B",
    "Medium": "#F2B824",
    "High": "#F28C28",
    "Very High": "#E5484D"
}


# =========================================================
# LOADERS
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)

    if "Exam_Score" in data.columns:
        data = data[data["Exam_Score"] <= 100].copy()

    return data.reset_index(drop=True)


# =========================================================
# MODEL / BUSINESS LOGIC
# =========================================================

def risk_category(probability):
    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Medium"
    if probability < 0.75:
        return "High"
    return "Very High"


def utility(data):
    return (
        BASE_WEIGHTS["ml_risk"] * data["Academic_Risk_Probability"]
        + BASE_WEIGHTS["attendance_risk"] * data["Attendance_Risk_Numeric"]
        + BASE_WEIGHTS["previous_performance_risk"] * data["Previous_Performance_Risk"]
    )


def key_factors(row, low_study_cutoff, low_previous_cutoff):
    factors = []

    if row["Attendance"] < 70:
        factors.append("Attendance")

    if row["Previous_Scores"] <= low_previous_cutoff:
        factors.append("Low Previous Scores")

    if row["Hours_Studied"] <= low_study_cutoff:
        factors.append("Low Study Hours")

    if str(row["Motivation_Level"]).lower() == "low":
        factors.append("Low Motivation")

    if str(row["Access_to_Resources"]).lower() == "low":
        factors.append("Limited Resources")

    if not factors:
        factors.append("Combined Model Factors")

    return ", ".join(factors[:3])


def recommended_action(row):
    actions = []

    if row["Attendance"] < 70:
        actions.append("Attendance Monitoring")

    if row["Previous_Scores"] < 65:
        actions.append("Tutoring")

    if str(row["Motivation_Level"]).lower() == "low":
        actions.append("Mentoring")

    if str(row["Access_to_Resources"]).lower() == "low":
        actions.append("Resource Support")

    if row["Academic_Risk_Probability"] >= 0.75:
        actions.append("Academic Review")

    if not actions:
        actions.append("Academic Review")

    return " + ".join(list(dict.fromkeys(actions))[:2])


def analyse_students(data, model):
    analysed = data.copy()

    analysed["Student_ID"] = [
        f"S{1001 + i:05d}" for i in range(len(analysed))
    ]

    probs = model.predict_proba(
        analysed[REQUIRED_FEATURES]
    )[:, 1]

    analysed["Academic_Risk_Probability"] = probs
    analysed["Risk_Category"] = [
        risk_category(p) for p in probs
    ]

    analysed["Attendance_Risk"] = analysed["Attendance"] < 70
    analysed["Attendance_Risk_Numeric"] = analysed["Attendance_Risk"].astype(int)

    analysed["Previous_Performance_Risk"] = (
        1 - analysed["Previous_Scores"] / 100
    ).clip(0, 1)

    analysed["Intervention_Utility"] = utility(analysed)

    analysed["Eligible_For_Intervention"] = (
        (analysed["Academic_Risk_Probability"] >= 0.50)
        | analysed["Attendance_Risk"]
    )

    study_cutoff = analysed["Hours_Studied"].quantile(0.25)
    previous_cutoff = analysed["Previous_Scores"].quantile(0.25)

    analysed["Key_Factors"] = analysed.apply(
        lambda r: key_factors(r, study_cutoff, previous_cutoff),
        axis=1
    )

    analysed["Recommended_Action"] = analysed.apply(
        recommended_action,
        axis=1
    )

    return analysed


def risk_factor_counts(data):
    study_cutoff = data["Hours_Studied"].quantile(0.25)
    previous_cutoff = data["Previous_Scores"].quantile(0.25)

    values = [
        ("Attendance Issues", int((data["Attendance"] < 70).sum())),
        ("Low Previous Scores", int((data["Previous_Scores"] <= previous_cutoff).sum())),
        ("Low Study Hours", int((data["Hours_Studied"] <= study_cutoff).sum())),
        ("Low Motivation", int(data["Motivation_Level"].astype(str).str.lower().eq("low").sum())),
        ("Limited Resources", int(data["Access_to_Resources"].astype(str).str.lower().eq("low").sum())),
    ]

    return pd.DataFrame(values, columns=["Risk Factor", "Students"])


def run_optimisation(eligible, capacity):
    data = (
        eligible
        .reset_index()
        .rename(columns={"index": "Student_Index"})
    )

    if capacity <= 0 or data.empty:
        return data.iloc[0:0].copy()

    utilities = data["Intervention_Utility"].to_numpy()
    n = len(data)

    result = milp(
        c=-utilities,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(
            np.ones((1, n)),
            lb=capacity,
            ub=capacity
        )
    )

    if not result.success:
        raise RuntimeError(result.message)

    selected = np.where(result.x > 0.5)[0]

    return (
        data.iloc[selected]
        .copy()
        .sort_values("Intervention_Utility", ascending=False)
    )


def strategy_comparison(eligible, capacity, trials=1000):
    capacity = min(capacity, len(eligible))

    rng = np.random.default_rng(42)
    idx = eligible.index.to_numpy()

    random_utility = []

    for _ in range(trials):
        chosen = rng.choice(idx, size=capacity, replace=False)
        random_utility.append(
            eligible.loc[chosen, "Intervention_Utility"].sum()
        )

    highest = (
        eligible
        .sort_values("Academic_Risk_Probability", ascending=False)
        .head(capacity)
    )

    optimised = run_optimisation(eligible, capacity)

    comparison = pd.DataFrame({
        "Strategy": [
            "Random Allocation",
            "Highest ML Risk",
            "Optimised Allocation"
        ],
        "Total Utility": [
            np.mean(random_utility),
            highest["Intervention_Utility"].sum(),
            optimised["Intervention_Utility"].sum()
        ]
    })

    return comparison, optimised


def support_mix(selected):
    categories = {
        "Tutoring": 0,
        "Attendance Monitoring": 0,
        "Mentoring": 0,
        "Resource Support": 0,
        "Academic Review": 0
    }

    for _, row in selected.iterrows():
        action = recommended_action(row)

        for key in categories:
            if key in action:
                categories[key] += 1

    result = pd.DataFrame({
        "Support Type": list(categories.keys()),
        "Students": list(categories.values())
    })

    return result[result["Students"] > 0]


# =========================================================
# CHART HELPERS
# =========================================================

def risk_bar_chart(data):
    counts = (
        data["Risk_Category"]
        .value_counts()
        .reindex(RISK_ORDER)
        .fillna(0)
        .reset_index()
    )

    counts.columns = ["Risk Level", "Students"]

    chart = (
        alt.Chart(counts)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
            size=42
        )
        .encode(
            x=alt.X(
                "Risk Level:N",
                sort=RISK_ORDER,
                axis=alt.Axis(
                    title=None,
                    labelAngle=0,
                    labelColor="#516176",
                    labelFontSize=11,
                    ticks=False,
                    domain=False
                )
            ),
            y=alt.Y(
                "Students:Q",
                axis=alt.Axis(
                    title=None,
                    grid=True,
                    gridColor="#E9EEF5",
                    gridOpacity=1,
                    labelColor="#6B7A90",
                    labelFontSize=10,
                    ticks=False,
                    domain=False
                )
            ),
            color=alt.Color(
                "Risk Level:N",
                scale=alt.Scale(
                    domain=RISK_ORDER,
                    range=[
                        RISK_COLORS["Low"],
                        RISK_COLORS["Medium"],
                        RISK_COLORS["High"],
                        RISK_COLORS["Very High"]
                    ]
                ),
                legend=None
            ),
            tooltip=["Risk Level", "Students"]
        )
        .properties(
            height=205,
            background="#FFFFFF"
        )
        .configure_view(stroke=None)
    )

    return chart

def risk_factor_donut(data):
    factors = (
        risk_factor_counts(data)
        .sort_values("Students", ascending=False)
        .reset_index(drop=True)
    )

    colors = [
        "#E5484D",
        "#F2B824",
        "#2F80ED",
        "#8B5CF6",
        "#A0AEC0"
    ]

    donut = (
        alt.Chart(factors)
        .mark_arc(
            innerRadius=46,
            outerRadius=78,
            stroke="#FFFFFF",
            strokeWidth=2
        )
        .encode(
            theta=alt.Theta("Students:Q"),
            color=alt.Color(
                "Risk Factor:N",
                scale=alt.Scale(
                    domain=factors["Risk Factor"].tolist(),
                    range=colors
                ),
                legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    columns=2,
                    title=None,
                    labelColor="#516176",
                    labelFontSize=9,
                    labelLimit=145,
                    symbolSize=75,
                    columnPadding=12,
                    rowPadding=5
                )
            ),
            tooltip=[
                alt.Tooltip("Risk Factor:N", title="Risk factor"),
                alt.Tooltip("Students:Q", title="Students")
            ]
        )
        .properties(
            height=250,
            background="#FFFFFF",
            padding={"top": 4, "left": 4, "right": 4, "bottom": 6}
        )
        .configure_view(stroke=None)
    )

    return donut

def support_bar_chart(data):
    if data.empty:
        return None

    display = data.copy()

    label_map = {
        "Attendance Monitoring": "Attendance",
        "Resource Support": "Resources",
        "Academic Review": "Academic Review",
        "Tutoring": "Tutoring",
        "Mentoring": "Mentoring"
    }

    display["Display Label"] = (
        display["Support Type"]
        .map(label_map)
        .fillna(display["Support Type"])
    )

    colors = [
        "#2F80ED",
        "#23B47E",
        "#8B5CF6",
        "#F59E42",
        "#94A3B8"
    ]

    domain = display["Support Type"].tolist()

    # Kept intentionally simple for compatibility with older
    # Altair/jsonschema versions.
    chart = (
        alt.Chart(display)
        .mark_bar(
            cornerRadiusEnd=5,
            size=24
        )
        .encode(
            y=alt.Y(
                "Display Label:N",
                sort="-x",
                title=None
            ),
            x=alt.X(
                "Students:Q",
                title=None
            ),
            color=alt.Color(
                "Support Type:N",
                scale=alt.Scale(
                    domain=domain,
                    range=colors[:len(domain)]
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip("Support Type:N", title="Support"),
                alt.Tooltip("Students:Q", title="Students")
            ]
        )
        .properties(
            height=205,
            background="#FFFFFF"
        )
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#516176",
            titleColor="#516176",
            gridColor="#E9EEF5",
            domain=False,
            ticks=False,
            labelFontSize=10
        )
    )

    return chart

def render_dashboard_student_table(data, max_rows=20):
    view = data.head(max_rows).copy()

    if view.empty:
        st.info("No students match the selected filters.")
        return

    def safe(value):
        return html_lib.escape(str(value))

    def badge_class(label):
        return {
            "Low": "risk-low",
            "Medium": "risk-medium",
            "High": "risk-high",
            "Very High": "risk-very-high",
            "On Track": "risk-low",
            "At Risk": "risk-very-high",
            "Very High Risk": "risk-medium"
        }.get(str(label), "risk-medium")

    def score_color(label):
        return {
            "Low": "#22A06B",
            "Medium": "#F2B824",
            "High": "#F28C28",
            "Very High": "#E5484D",
            "On Track": "#22A06B",
            "At Risk": "#E5484D",
            "Very High Risk": "#F2B824"
        }.get(str(label), "#2F80ED")

    show_status = "Display_Status" in view.columns
    category_header = "Status" if show_status else "Risk Level"

    rows = []

    for _, row in view.iterrows():
        underlying_risk = str(row["Risk_Category"])
        display_label = (
            str(row["Display_Status"])
            if show_status
            else underlying_risk
        )

        score = float(row["Academic_Risk_Probability"]) * 100
        attendance = float(row["Attendance"])
        previous = float(row["Previous_Scores"])

        rows.append(
            "<tr>"
            f"<td><strong>{safe(row['Student_ID'])}</strong></td>"
            f"<td><span class='risk-badge {badge_class(display_label)}'>{safe(display_label)}</span></td>"
            "<td>"
            f"<span class='score-track'><span class='score-fill' "
            f"style='display:block;width:{max(0,min(100,score)):.1f}%;"
            f"background:{score_color(display_label)};'></span></span>"
            f"<strong>{score:.1f}%</strong>"
            "</td>"
            f"<td>{attendance:.0f}%</td>"
            f"<td>{previous:.0f}</td>"
            f"<td>{safe(row['Key_Factors'])}</td>"
            f"<td>{safe(row['Recommended_Action'])}</td>"
            "</tr>"
        )

    html = (
        "<div class='ews-table-wrap'>"
        "<table class='ews-table'>"
        "<thead><tr>"
        "<th>Student ID</th>"
        f"<th>{category_header}</th>"
        "<th>Risk Score</th>"
        "<th>Attendance</th>"
        "<th>Previous Score</th>"
        "<th>Key Factors</th>"
        "<th>Recommended Action</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )

    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# LOAD PROJECT DATA
# =========================================================

try:
    model = load_model()
    raw_data = load_data()
except FileNotFoundError as exc:
    st.error(f"Missing required file: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the project: {exc}")
    st.stop()

missing = [c for c in REQUIRED_FEATURES if c not in raw_data.columns]

if missing:
    st.error("Missing required dataset columns: " + ", ".join(missing))
    st.stop()

analysed = analyse_students(raw_data, model)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <div style="padding:0.4rem 0.35rem 1.05rem 0.35rem;">
        <div style="display:flex;align-items:center;gap:0.65rem;">
            <div style="
                width:34px;height:34px;border-radius:9px;
                background:#1E73E8;display:flex;align-items:center;
                justify-content:center;font-size:17px;font-weight:800;">
                E
            </div>
            <div>
                <div style="font-size:1.25rem;font-weight:800;">EWS</div>
                <div style="font-size:0.7rem;color:#AFC0D5;">Early Warning System</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

sidebar_options = {
    "🏠  Dashboard": "Dashboard",
    "👥  Students": "Students",
    "⚠️  Risk Analysis": "Risk Analysis",
    "⚙️  Resource Allocation": "Resource Allocation",
    "📊  Reports": "Reports",
    "🔧  Settings": "Settings"
}

page_label = st.sidebar.radio(
    "Page",
    list(sidebar_options.keys()),
    label_visibility="collapsed"
)

page = sidebar_options[page_label]

st.sidebar.markdown(
    """
    <div style="
        position:fixed;
        bottom:20px;
        left:20px;
        color:#B8C6D8;
        font-size:0.75rem;
        line-height:1.4;">
        <div style="font-weight:700;color:white;">James</div>
        <div>Administrator</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TOP BAR
# =========================================================

st.markdown(
    f"""
    <div class="topbar">
        <div class="search-box">⌕ &nbsp; Search student by ID...</div>
        <div class="top-date">{date.today().strftime("%a, %d %b %Y")}</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-title">Welcome back, James</div>
            <div class="welcome-sub">Here's an overview of the Early Warning System</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_students = len(analysed)

    # Mutually exclusive dashboard status groups.
    # On Track: not eligible under the intervention rule.
    # Very High Risk: ML risk >= 75%.
    # At Risk: intervention-eligible students who are not already Very High Risk.
    on_track_mask = ~analysed["Eligible_For_Intervention"]
    very_high_mask = analysed["Academic_Risk_Probability"] >= 0.75
    at_risk_mask = (
        analysed["Eligible_For_Intervention"]
        & ~very_high_mask
    )

    on_track = int(on_track_mask.sum())
    very_high = int(very_high_mask.sum())
    at_risk = int(at_risk_mask.sum())

    at_risk_pct = at_risk / total_students * 100
    very_high_pct = very_high / total_students * 100
    on_track_pct = on_track / total_students * 100

    if "dashboard_segment" not in st.session_state:
        st.session_state.dashboard_segment = None

    def select_dashboard_segment(segment):
        st.session_state.dashboard_segment = segment

    def clear_dashboard_segment():
        st.session_state.dashboard_segment = None

    a, b, c, d = st.columns(4)

    with a:
        st.button(
            f"●  Total Students\n\n{total_students:,}\n\nStudent records analysed",
            key="card_total",
            use_container_width=True,
            on_click=select_dashboard_segment,
            args=("total",)
        )

    with b:
        st.button(
            f"❗  At Risk Students\n\n{at_risk:,}\n\n{at_risk_pct:.1f}% of total",
            key="card_at_risk",
            use_container_width=True,
            on_click=select_dashboard_segment,
            args=("at-risk",)
        )

    with c:
        st.button(
            f"▲  Very High Risk\n\n{very_high:,}\n\n{very_high_pct:.1f}% of total",
            key="card_very_high",
            use_container_width=True,
            on_click=select_dashboard_segment,
            args=("very-high",)
        )

    with d:
        st.button(
            f"✓  On Track\n\n{on_track:,}\n\n{on_track_pct:.1f}% of total",
            key="card_on_track",
            use_container_width=True,
            on_click=select_dashboard_segment,
            args=("on-track",)
        )

    selected_segment = st.session_state.dashboard_segment

    if selected_segment:
        segment_config = {
            "total": {
                "title": "Total Students",
                "description": "All student records currently analysed by the Early Warning System.",
                "tone": "segment-blue",
                "status": None,
                "data": analysed.copy()
            },
            "at-risk": {
                "title": "At Risk Students",
                "description": "Intervention-eligible students who are not already in the Very High Risk group.",
                "tone": "segment-red",
                "status": "At Risk",
                "data": analysed[at_risk_mask].copy()
            },
            "very-high": {
                "title": "Very High Risk Students",
                "description": "Students whose predicted academic-risk probability is 75% or higher.",
                "tone": "segment-yellow",
                "status": "Very High Risk",
                "data": analysed[very_high_mask].copy()
            },
            "on-track": {
                "title": "On Track Students",
                "description": "Students who do not currently meet the intervention eligibility rules.",
                "tone": "segment-green",
                "status": "On Track",
                "data": analysed[on_track_mask].copy()
            }
        }

        segment = segment_config[selected_segment]
        selected_data = segment["data"].copy()

        if segment["status"] is not None:
            selected_data["Display_Status"] = segment["status"]

        st.markdown(
            f"""
            <div class="selected-segment-panel {segment["tone"]}">
                <div class="selected-segment-title">{segment["title"]}</div>
                <div class="selected-segment-sub">
                    {segment["description"]} &nbsp;·&nbsp; {len(selected_data):,} students
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        filter_a, filter_b, filter_c, filter_d = st.columns([1, 1.2, 2, 0.8])

        segment_risk = filter_a.selectbox(
            "ML Risk Level",
            ["All", "Low", "Medium", "High", "Very High"],
            key=f"segment_risk_{selected_segment}"
        )

        segment_attendance = filter_b.selectbox(
            "Attendance",
            ["All", "Attendance Risk", "No Attendance Risk"],
            key=f"segment_attendance_{selected_segment}"
        )

        segment_search = filter_c.text_input(
            "Search Student ID",
            placeholder="e.g. S01001",
            key=f"segment_search_{selected_segment}"
        )

        segment_rows = filter_d.selectbox(
            "Rows",
            [10, 25, 50, 100],
            index=1,
            key=f"segment_rows_{selected_segment}"
        )

        if segment_risk != "All":
            selected_data = selected_data[
                selected_data["Risk_Category"] == segment_risk
            ]

        if segment_attendance == "Attendance Risk":
            selected_data = selected_data[selected_data["Attendance_Risk"]]
        elif segment_attendance == "No Attendance Risk":
            selected_data = selected_data[~selected_data["Attendance_Risk"]]

        if segment_search.strip():
            selected_data = selected_data[
                selected_data["Student_ID"].str.contains(
                    segment_search.strip(),
                    case=False,
                    na=False
                )
            ]

        selected_data = selected_data.sort_values(
            "Academic_Risk_Probability",
            ascending=False
        )

        render_dashboard_student_table(
            selected_data,
            max_rows=segment_rows
        )

    st.write("")

    # Broad dashboard charts
    left, middle, right = st.columns([1.02, 1.08, 1.18], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="dashboard-chart-title">Risk Level Overview</div>'
                '<div class="dashboard-chart-sub">Model risk across all students</div>',
                unsafe_allow_html=True
            )
            st.altair_chart(
                risk_bar_chart(analysed),
                use_container_width=True,
                theme=None
            )

    with middle:
        with st.container(border=True):
            st.markdown(
                '<div class="dashboard-chart-title">Risk Factors (Top 5)</div>'
                '<div class="dashboard-chart-sub">Most common measurable warning indicators</div>',
                unsafe_allow_html=True
            )
            donut = risk_factor_donut(analysed)
            st.altair_chart(
                donut,
                use_container_width=True,
                theme=None
            )

    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="dashboard-chart-title">Resource Allocation (Optimised)</div>'
                '<div class="dashboard-chart-sub">Support recommended for the top 10 priorities</div>',
                unsafe_allow_html=True
            )

            eligible_for_dash = analysed[
                analysed["Eligible_For_Intervention"]
            ].copy()

            cap = min(10, len(eligible_for_dash))

            if cap:
                dash_selected = run_optimisation(eligible_for_dash, cap)
                dash_support = support_mix(dash_selected)
                chart = support_bar_chart(dash_support)

                if chart is not None:
                    st.altair_chart(
                        chart,
                        use_container_width=True,
                        theme=None
                    )
            else:
                st.info("No eligible students.")

    # Students at risk
    st.markdown("### Students at Risk")

    f1, f2, f3, f4 = st.columns([1, 1, 1.5, 0.8])

    selected_risk = f1.selectbox(
        "Risk Level",
        ["All", "Low", "Medium", "High", "Very High"],
        key="dash_risk"
    )

    selected_attendance = f2.selectbox(
        "Attendance",
        ["All", "Attendance Risk", "No Attendance Risk"],
        key="dash_att"
    )

    search_id = f3.text_input(
        "Search Student ID",
        placeholder="e.g. S01001",
        key="dash_search"
    )

    sort_by = f4.selectbox(
        "Sort By",
        ["Risk Score", "Utility"],
        key="dash_sort"
    )

    table = analysed[
        analysed["Eligible_For_Intervention"]
    ].copy()

    if selected_risk != "All":
        table = table[table["Risk_Category"] == selected_risk]

    if selected_attendance == "Attendance Risk":
        table = table[table["Attendance_Risk"]]
    elif selected_attendance == "No Attendance Risk":
        table = table[~table["Attendance_Risk"]]

    if search_id.strip():
        table = table[
            table["Student_ID"].str.contains(
                search_id.strip(),
                case=False,
                na=False
            )
        ]

    sort_col = (
        "Academic_Risk_Probability"
        if sort_by == "Risk Score"
        else "Intervention_Utility"
    )

    table = table.sort_values(sort_col, ascending=False).head(20)

    table_display = table[[
        "Student_ID",
        "Risk_Category",
        "Academic_Risk_Probability",
        "Attendance",
        "Previous_Scores",
        "Key_Factors",
        "Recommended_Action"
    ]].copy()

    table_display["Academic_Risk_Probability"] = (
        table_display["Academic_Risk_Probability"] * 100
    ).round(1)

    table_display = table_display.rename(columns={
        "Risk_Category": "Risk Level",
        "Academic_Risk_Probability": "Risk Score",
        "Previous_Scores": "Previous Score",
        "Key_Factors": "Key Factors",
        "Recommended_Action": "Recommended Action"
    })

    render_dashboard_student_table(
        table,
        max_rows=20
    )

    # Bottom overview row
    st.write("")
    bottom_left, bottom_right = st.columns([1, 1], gap="large")

    with bottom_left:
        with st.container(border=True):
            st.markdown(
                '<div class="dashboard-chart-title">Resource Allocation by Risk Level</div>'
                '<div class="dashboard-chart-sub">Risk composition of the top 50 intervention priorities</div>',
                unsafe_allow_html=True
            )

            eligible = analysed[
                analysed["Eligible_For_Intervention"]
            ].copy()

            top50 = eligible.nlargest(
                min(50, len(eligible)),
                "Intervention_Utility"
            )

            allocation = (
                top50["Risk_Category"]
                .value_counts()
                .reindex(RISK_ORDER)
                .fillna(0)
                .reset_index()
            )
            allocation.columns = ["Risk Level", "Students"]

            allocation_chart = (
                alt.Chart(allocation)
                .mark_arc(
                    innerRadius=52,
                    outerRadius=92,
                    stroke="#FFFFFF",
                    strokeWidth=2
                )
                .encode(
                    theta="Students:Q",
                    color=alt.Color(
                        "Risk Level:N",
                        scale=alt.Scale(
                            domain=RISK_ORDER,
                            range=[
                                RISK_COLORS["Low"],
                                RISK_COLORS["Medium"],
                                RISK_COLORS["High"],
                                RISK_COLORS["Very High"]
                            ]
                        ),
                        legend=alt.Legend(
                            orient="bottom",
                            title=None,
                            labelColor="#516176"
                        )
                    ),
                    tooltip=["Risk Level", "Students"]
                )
                .properties(
                    height=245,
                    background="#FFFFFF"
                )
                .configure_view(stroke=None)
            )

            st.altair_chart(
                allocation_chart,
                use_container_width=True,
                theme=None
            )

    with bottom_right:
        with st.container(border=True):
            st.markdown(
                '<div class="dashboard-chart-title">Student Details</div>'
                '<div class="dashboard-chart-sub">Inspect one student record</div>',
                unsafe_allow_html=True
            )

            detail_ids = table["Student_ID"].tolist()

            if not detail_ids:
                detail_ids = analysed["Student_ID"].head(50).tolist()

            selected_id = st.selectbox(
                "Student",
                detail_ids,
                label_visibility="collapsed",
                key="dashboard_student_detail"
            )

            row = analysed[
                analysed["Student_ID"] == selected_id
            ].iloc[0]

            risk_label = row["Risk_Category"]
            pill_class = {
                "Low": "pill-low",
                "Medium": "pill-medium",
                "High": "pill-high",
                "Very High": "pill-very-high"
            }[risk_label]

            detail_html = (
                "<div class='student-detail'>"
                "<div style='display:flex;justify-content:space-between;align-items:center;gap:1rem;'>"
                f"<div class='student-id'>{html_lib.escape(str(row['Student_ID']))}</div>"
                f"<span class='risk-pill {pill_class}'>{html_lib.escape(str(risk_label))} Risk</span>"
                "</div>"
                "<div class='student-detail-grid'>"
                "<div class='student-detail-item'>"
                "<div class='detail-label'>Risk Score</div>"
                f"<div class='detail-value'>{row['Academic_Risk_Probability']*100:.1f}%</div>"
                "</div>"
                "<div class='student-detail-item'>"
                "<div class='detail-label'>Attendance</div>"
                f"<div class='detail-value'>{row['Attendance']:.0f}%</div>"
                "</div>"
                "<div class='student-detail-item'>"
                "<div class='detail-label'>Previous Score</div>"
                f"<div class='detail-value'>{row['Previous_Scores']:.0f}</div>"
                "</div>"
                "<div class='student-detail-item'>"
                "<div class='detail-label'>Hours Studied</div>"
                f"<div class='detail-value'>{row['Hours_Studied']:.0f}</div>"
                "</div>"
                "<div class='student-detail-item' style='grid-column:1/-1;'>"
                "<div class='detail-label'>Key Risk Factors</div>"
                f"<div class='detail-value'>{html_lib.escape(str(row['Key_Factors']))}</div>"
                "</div>"
                "<div class='student-detail-item' style='grid-column:1/-1;'>"
                "<div class='detail-label'>Recommended Action</div>"
                f"<div class='detail-value'>{html_lib.escape(str(row['Recommended_Action']))}</div>"
                "</div>"
                "</div>"
                "</div>"
            )

            st.markdown(detail_html, unsafe_allow_html=True)


# =========================================================
# STUDENTS
# =========================================================

elif page == "Students":

    st.markdown("## Students")

    c1, c2, c3 = st.columns([2, 1, 1])

    search = c1.text_input(
        "Search Student ID",
        placeholder="e.g. S01001"
    )

    risk = c2.selectbox(
        "Risk Level",
        ["All", "Low", "Medium", "High", "Very High"]
    )

    eligibility = c3.selectbox(
        "Status",
        ["All", "At Risk", "On Track"]
    )

    student_data = analysed.copy()

    if search.strip():
        student_data = student_data[
            student_data["Student_ID"].str.contains(
                search.strip(),
                case=False,
                na=False
            )
        ]

    if risk != "All":
        student_data = student_data[
            student_data["Risk_Category"] == risk
        ]

    if eligibility == "At Risk":
        student_data = student_data[
            student_data["Eligible_For_Intervention"]
        ]
    elif eligibility == "On Track":
        student_data = student_data[
            ~student_data["Eligible_For_Intervention"]
        ]

    out = student_data[[
        "Student_ID",
        "Risk_Category",
        "Academic_Risk_Probability",
        "Attendance",
        "Hours_Studied",
        "Previous_Scores",
        "Attendance_Risk",
        "Key_Factors",
        "Recommended_Action"
    ]].copy()

    out["Academic_Risk_Probability"] = (
        out["Academic_Risk_Probability"] * 100
    ).round(1)

    out = out.rename(columns={
        "Risk_Category": "Risk Level",
        "Academic_Risk_Probability": "Risk Score",
        "Hours_Studied": "Hours Studied",
        "Previous_Scores": "Previous Score",
        "Attendance_Risk": "Attendance Risk",
        "Key_Factors": "Key Factors",
        "Recommended_Action": "Recommended Action"
    })

    st.dataframe(
        out,
        hide_index=True,
        use_container_width=True,
        height=650,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            )
        }
    )


# =========================================================
# RISK ANALYSIS
# =========================================================

elif page == "Risk Analysis":

    st.markdown("## Risk Analysis")

    total = len(analysed)
    attendance_risk = int(analysed["Attendance_Risk"].sum())
    ml_risk = int((analysed["Academic_Risk_Probability"] >= 0.50).sum())
    very_high = int((analysed["Academic_Risk_Probability"] >= 0.75).sum())

    a, b, c = st.columns(3)

    a.metric("Attendance Risk", f"{attendance_risk:,}")
    b.metric("ML Risk ≥ 50%", f"{ml_risk:,}")
    c.metric("Very High Risk", f"{very_high:,}")

    left, right = st.columns(2)

    with left:
        st.markdown("### Risk Levels")
        st.altair_chart(
            risk_bar_chart(analysed),
            use_container_width=True,
            theme=None
        )

    with right:
        st.markdown("### Top Risk Factors")
        donut = risk_factor_donut(analysed)
        st.altair_chart(
            donut,
            use_container_width=True,
            theme=None
        )

    st.markdown("### Risk Factor Counts")
    st.dataframe(
        risk_factor_counts(analysed)
        .sort_values("Students", ascending=False),
        hide_index=True,
        use_container_width=True
    )


# =========================================================
# RESOURCE ALLOCATION
# =========================================================

elif page == "Resource Allocation":

    st.markdown("## Resource Allocation")

    eligible = analysed[
        analysed["Eligible_For_Intervention"]
    ].copy()

    if eligible.empty:
        st.success("No students currently meet the intervention criteria.")
        st.stop()

    top1, top2 = st.columns([1, 3])

    capacity = top1.number_input(
        "Available Intervention Places",
        min_value=1,
        max_value=len(eligible),
        value=min(10, len(eligible)),
        step=1
    )

    top2.markdown(
        f"""
        <div class="summary-card card-blue" style="min-height:105px;">
            <div class="summary-label">Eligible Students</div>
            <div class="summary-value">{len(eligible):,}</div>
            <div class="summary-sub">ML risk ≥ 50% or attendance below 70%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "Optimise Resource Allocation",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Optimising available intervention places..."):
            comparison, optimised = strategy_comparison(
                eligible,
                int(capacity)
            )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Strategy Comparison")

            strategy_chart = (
                alt.Chart(comparison)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X(
                        "Strategy:N",
                        axis=alt.Axis(title=None, labelAngle=0)
                    ),
                    y=alt.Y(
                        "Total Utility:Q",
                        axis=alt.Axis(title=None, gridColor="#EEF2F7")
                    ),
                    color=alt.Color(
                        "Strategy:N",
                        scale=alt.Scale(
                            domain=[
                                "Random Allocation",
                                "Highest ML Risk",
                                "Optimised Allocation"
                            ],
                            range=[
                                "#A0AEC0",
                                "#F2B824",
                                "#2F80ED"
                            ]
                        ),
                        legend=None
                    ),
                    tooltip=[
                        "Strategy",
                        alt.Tooltip("Total Utility:Q", format=".3f")
                    ]
                )
                .properties(height=300)
                .configure_view(stroke=None)
            )

            st.altair_chart(
                strategy_chart,
                use_container_width=True,
                theme=None
            )

        with c2:
            st.markdown("### Recommended Support Mix")

            mix = support_mix(optimised)
            chart = support_bar_chart(mix)

            if chart is not None:
                st.altair_chart(
                    chart,
                    use_container_width=True,
                    theme=None
                )

        st.markdown("### Prioritised Students")

        priority = optimised[[
            "Student_ID",
            "Risk_Category",
            "Academic_Risk_Probability",
            "Attendance",
            "Previous_Scores",
            "Key_Factors",
            "Recommended_Action",
            "Intervention_Utility"
        ]].copy()

        priority["Academic_Risk_Probability"] = (
            priority["Academic_Risk_Probability"] * 100
        ).round(1)

        priority["Intervention_Utility"] = (
            priority["Intervention_Utility"]
        ).round(4)

        priority = priority.rename(columns={
            "Risk_Category": "Risk Level",
            "Academic_Risk_Probability": "Risk Score",
            "Previous_Scores": "Previous Score",
            "Key_Factors": "Key Factors",
            "Recommended_Action": "Recommended Action",
            "Intervention_Utility": "Utility Score"
        })

        st.dataframe(
            priority,
            hide_index=True,
            use_container_width=True
        )


# =========================================================
# REPORTS
# =========================================================

elif page == "Reports":

    st.markdown("## Reports")

    st.markdown("### Download Student Risk Report")

    report = analysed.copy()

    st.download_button(
        "Download Full Student Risk Report",
        report.to_csv(index=False).encode("utf-8"),
        file_name="student_risk_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("### Model Evaluation")

    try:
        metrics = pd.read_csv(HELDOUT_METRICS_FILE)

        if not metrics.empty:
            row = metrics.iloc[0]
            a, b, c, d, e = st.columns(5)
            a.metric("Accuracy", f"{row['Accuracy']:.3f}")
            b.metric("Precision", f"{row['Precision']:.3f}")
            c.metric("Recall", f"{row['Recall']:.3f}")
            d.metric("F1", f"{row['F1']:.3f}")
            e.metric("ROC-AUC", f"{row['ROC_AUC']:.3f}")

    except FileNotFoundError:
        st.warning("Held-out metrics file was not found.")

    try:
        cv = pd.read_csv(CV_RESULTS_FILE)
        st.markdown("### Cross-Validation Model Comparison")
        st.dataframe(
            cv.round(4),
            hide_index=True,
            use_container_width=True
        )
    except FileNotFoundError:
        pass


# =========================================================
# SETTINGS
# =========================================================

elif page == "Settings":

    st.markdown("## Settings")

    st.markdown("### Current Decision Rules")

    settings = pd.DataFrame({
        "Setting": [
            "Attendance Risk Threshold",
            "ML Intervention Eligibility",
            "Low Risk",
            "Medium Risk",
            "High Risk",
            "Very High Risk",
            "ML Utility Weight",
            "Attendance Utility Weight",
            "Previous Performance Utility Weight"
        ],
        "Value": [
            "< 70%",
            "≥ 50%",
            "0%–24.9%",
            "25%–49.9%",
            "50%–74.9%",
            "75%–100%",
            "60%",
            "25%",
            "15%"
        ]
    })

    st.dataframe(
        settings,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "These settings reflect the current prototype methodology. "
        "Changing them would require re-evaluating the system."
    )
