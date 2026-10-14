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

    risk = row["Academic_Risk_Probability"]

    # Specific student problems
    if row["Attendance"] < 70:
        actions.append("Attendance Monitoring")

    if row["Previous_Scores"] < 65:
        actions.append("Tutoring")

    if str(row["Motivation_Level"]).lower() == "low":
        actions.append("Mentoring")

    if str(row["Access_to_Resources"]).lower() == "low":
        actions.append("Resource Support")

    # Very high risk
    if risk >= 0.75:
        actions.insert(0, "Academic Review")

    # High risk but no specific problem found
    elif risk >= 0.50 and not actions:
        actions.append("Academic Support Review")

    # Medium risk with no specific problem
    elif risk >= 0.25 and not actions:
        return "Continue Monitoring"

    # Low risk with no specific problem
    elif risk < 0.25 and not actions:
        return "No Intervention Required"

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

def intervention_cost(row):
    """
    Estimated resource cost for supporting a student.
    Higher-cost interventions consume more of the available budget.
    """

    cost = 1

    if row["Attendance"] < 70:
        cost += 1

    if row["Previous_Scores"] < 65:
        cost += 2

    if str(row["Motivation_Level"]).lower() == "low":
        cost += 1

    if str(row["Access_to_Resources"]).lower() == "low":
        cost += 2

    if row["Academic_Risk_Probability"] >= 0.75:
        cost += 2

    return cost
def run_optimisation(eligible, capacity, budget=None):
    data = (
        eligible
        .reset_index()
        .rename(columns={"index": "Student_Index"})
    )

    if capacity <= 0 or data.empty:
        return data.iloc[0:0].copy()

    # Calculate intervention cost for each student
    data["Intervention_Cost"] = data.apply(
        intervention_cost,
        axis=1
    )

    utilities = data["Intervention_Utility"].to_numpy()
    costs = data["Intervention_Cost"].to_numpy()

    n = len(data)

    # Do not allow capacity to exceed eligible students
    capacity = min(capacity, n)

    constraints = [
        # Maximum number of students that can receive support
        LinearConstraint(
            np.ones((1, n)),
            lb=0,
            ub=capacity
        )
    ]

    # Optional resource budget
    if budget is not None:
        constraints.append(
            LinearConstraint(
                costs.reshape(1, -1),
                lb=0,
                ub=budget
            )
        )

    result = milp(
        c=-utilities,
        integrality=np.ones(n),
        bounds=Bounds(
            np.zeros(n),
            np.ones(n)
        ),
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(result.message)

    selected = np.where(result.x > 0.5)[0]

    return (
        data.iloc[selected]
        .copy()
        .sort_values(
            "Intervention_Utility",
            ascending=False
        )
    )
def strategy_comparison(eligible, capacity, budget=None, trials=1000):
    data = eligible.copy()

    # Calculate intervention cost for every eligible student
    data["Intervention_Cost"] = data.apply(
        intervention_cost,
        axis=1
    )

    capacity = min(capacity, len(data))

    # -------------------------
    # Random allocation
    # -------------------------
    rng = np.random.default_rng(42)
    random_utility = []

    for _ in range(trials):
        shuffled = data.sample(
            frac=1,
            random_state=int(rng.integers(0, 1_000_000))
        )

        selected = []
        total_cost = 0

        for idx, row in shuffled.iterrows():
            cost = row["Intervention_Cost"]

            if len(selected) >= capacity:
                break

            if budget is None or total_cost + cost <= budget:
                selected.append(idx)
                total_cost += cost

        random_utility.append(
            data.loc[selected, "Intervention_Utility"].sum()
            if selected else 0
        )

    # -------------------------
    # Highest ML risk
    # -------------------------
    risk_sorted = data.sort_values(
        "Academic_Risk_Probability",
        ascending=False
    )

    highest_indices = []
    highest_cost = 0

    for idx, row in risk_sorted.iterrows():
        cost = row["Intervention_Cost"]

        if len(highest_indices) >= capacity:
            break

        if budget is None or highest_cost + cost <= budget:
            highest_indices.append(idx)
            highest_cost += cost

    highest = data.loc[highest_indices].copy()

    # -------------------------
    # Optimised allocation
    # -------------------------
    optimised = run_optimisation(
        data,
        capacity,
        budget=budget
    )

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
        "Academic Review": 0,
        "Academic Support Review": 0
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


# =========================================================
# SESSION STUDENTS
# =========================================================

if "added_students" not in st.session_state:
    st.session_state.added_students = []


# Add manually created students to the original dataset
if st.session_state.added_students:

    added_students_df = pd.DataFrame(
        st.session_state.added_students
    )

    combined_data = pd.concat(
        [raw_data, added_students_df],
        ignore_index=True,
        sort=False
    )

else:
    combined_data = raw_data.copy()


analysed = analyse_students(
    combined_data,
    model
)


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

search_col, date_col = st.columns([4, 1])

with search_col:
    top_search = st.text_input(
        "Search student by ID",
        placeholder="Search student by ID... e.g. S01001",
        label_visibility="collapsed",
        key="top_student_search"
    )

with date_col:
    st.markdown(
        f"""
        <div style="
            text-align:right;
            padding-top:0.55rem;
            color:#53657A;
            font-size:0.82rem;
            font-weight:600;
        ">
            {date.today().strftime("%a, %d %b %Y")}
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# STUDENT ID SEARCH
# =========================================================

if top_search.strip():

    student_id = top_search.strip().upper()

    result = analysed[
        analysed["Student_ID"].str.upper() == student_id
    ]

    if result.empty:
        st.warning(f"No student found with ID {student_id}")

    else:
        student = result.iloc[0]

        st.markdown(
            f"### Student Profile — {student['Student_ID']}"
        )

        # First row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Risk Level",
                student["Risk_Category"]
            )

        with col2:
            st.metric(
                "Risk Score",
                f"{student['Academic_Risk_Probability'] * 100:.1f}%"
            )

        with col3:
            st.metric(
                "Attendance",
                f"{student['Attendance']:.0f}%"
            )

        with col4:
            st.metric(
                "Previous Score",
                f"{student['Previous_Scores']:.0f}"
            )

        # Second row
        col5, col6, col7 = st.columns(3)

        with col5:
            st.metric(
                "Study Hours",
                f"{student['Hours_Studied']:.0f}"
            )

        with col6:
            intervention_status = (
                "Eligible"
                if student["Eligible_For_Intervention"]
                else "Not Required"
            )

            st.metric(
                "Intervention Status",
                intervention_status
            )

        with col7:
            st.metric(
                "Priority Score",
                f"{student['Intervention_Utility']:.2f}"
            )

        # Student support information
        st.markdown("#### Student Support Information")

        st.write(
            f"**Key Factors:** {student['Key_Factors']}"
        )

        st.write(
            f"**Recommended Action:** {student['Recommended_Action']}"
        )
# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-title">Student Early Warning System</div>
            <div class="welcome-sub">
                Quick overview of student academic support and intervention needs
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # DASHBOARD GROUPS
    # -----------------------------------------------------

    total_students = len(analysed)

    on_track_mask = ~analysed["Eligible_For_Intervention"]

    urgent_mask = (
        analysed["Academic_Risk_Probability"] >= 0.75
    )

    need_support_mask = (
        analysed["Eligible_For_Intervention"]
        & ~urgent_mask
    )

    on_track = int(on_track_mask.sum())
    urgent = int(urgent_mask.sum())
    need_support = int(need_support_mask.sum())

    need_support_pct = (
        need_support / total_students * 100
        if total_students else 0
    )

    urgent_pct = (
        urgent / total_students * 100
        if total_students else 0
    )

    on_track_pct = (
        on_track / total_students * 100
        if total_students else 0
    )

    # -----------------------------------------------------
    # SELECTED DASHBOARD VIEW
    # -----------------------------------------------------

    if "dashboard_segment" not in st.session_state:
        st.session_state.dashboard_segment = "total"

    def select_dashboard_segment(segment):
        st.session_state.dashboard_segment = segment

    # -----------------------------------------------------
    # SUMMARY CARDS
    # -----------------------------------------------------

    card1, card2, card3, card4 = st.columns(4)

    with card1:
        st.button(
            f"●  All Students\n\n"
            f"{total_students:,}\n\n"
            f"Student records",
            key="card_total",
            use_container_width=True,
            on_click=select_dashboard_segment,
            args=("total",)
        )

    with card2:
        st.button(
            f"❗  Need Support\n\n"
            f"{need_support:,}\n\n"
            f"{need_support_pct:.1f}% of students",
            key="card_at_risk",
            use_container_width=True,
            on_click=select_dashboard_segment,
            args=("need-support",)
        )

    with card3:
        st.button(
            f"▲  Urgent\n\n"
            f"{urgent:,}\n\n"
            f"{urgent_pct:.1f}% of students",
            key="card_very_high",
            use_container_width=True,
            on_click=select_dashboard_segment,
            args=("urgent",)
        )

    with card4:
        st.button(
            f"✓  On Track\n\n"
            f"{on_track:,}\n\n"
            f"{on_track_pct:.1f}% of students",
            key="card_on_track",
            use_container_width=True,
            on_click=select_dashboard_segment,
            args=("on-track",)
        )

    st.write("")

    selected_segment = st.session_state.dashboard_segment

    # =====================================================
    # ALL STUDENTS
    # =====================================================

    if selected_segment == "total":

        st.markdown("### Cohort Overview")

        st.caption(
            "A general overview of attendance and academic activity "
            "across the whole student population."
        )

        avg_attendance = analysed["Attendance"].mean()
        avg_previous = analysed["Previous_Scores"].mean()
        avg_study = analysed["Hours_Studied"].mean()

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Average Attendance",
            f"{avg_attendance:.1f}%"
        )

        metric2.metric(
            "Average Previous Score",
            f"{avg_previous:.1f}"
        )

        metric3.metric(
            "Average Study Hours",
            f"{avg_study:.1f} hrs"
        )

        st.write("")

        chart_left, chart_right = st.columns(
            [1.35, 1],
            gap="large"
        )

        # Attendance distribution
        with chart_left:

            with st.container(border=True):

                st.markdown("#### Attendance Distribution")

                st.caption(
                    "How attendance is distributed across all students"
                )

                attendance_chart = (
                    alt.Chart(analysed)
                    .mark_bar(
                        cornerRadiusTopLeft=4,
                        cornerRadiusTopRight=4
                    )
                    .encode(
                        x=alt.X(
                            "Attendance:Q",
                            bin=alt.Bin(step=5),
                            title="Attendance (%)"
                        ),
                        y=alt.Y(
                            "count():Q",
                            title="Number of Students"
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "count():Q",
                                title="Students"
                            )
                        ]
                    )
                    .properties(
                        height=280,
                        background="#FFFFFF"
                    )
                    .configure_view(
                        stroke=None
                    )
                )

                st.altair_chart(
                    attendance_chart,
                    use_container_width=True,
                    theme=None
                )

        # Previous score distribution
        with chart_right:

            with st.container(border=True):

                st.markdown("#### Previous Score Distribution")

                st.caption(
                    "Previous academic performance across the cohort"
                )

                previous_chart = (
                    alt.Chart(analysed)
                    .mark_bar(
                        cornerRadiusTopLeft=4,
                        cornerRadiusTopRight=4
                    )
                    .encode(
                        x=alt.X(
                            "Previous_Scores:Q",
                            bin=alt.Bin(step=5),
                            title="Previous Score"
                        ),
                        y=alt.Y(
                            "count():Q",
                            title="Students"
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "count():Q",
                                title="Students"
                            )
                        ]
                    )
                    .properties(
                        height=280,
                        background="#FFFFFF"
                    )
                    .configure_view(
                        stroke=None
                    )
                )

                st.altair_chart(
                    previous_chart,
                    use_container_width=True,
                    theme=None
                )

    # =====================================================
    # NEED SUPPORT
    # =====================================================

    elif selected_segment == "need-support":

        support_students = analysed[
            need_support_mask
        ].copy()

        st.markdown("### Students Needing Support")

        st.caption(
            "Students who currently meet the intervention criteria "
            "but are not in the urgent group."
        )

        support1, support2, support3 = st.columns(3)

        support1.metric(
            "Students Needing Support",
            f"{len(support_students):,}"
        )

        attendance_support = int(
            support_students["Attendance_Risk"].sum()
        )

        support2.metric(
            "Attendance Concerns",
            f"{attendance_support:,}"
        )

        avg_support_risk = (
            support_students[
                "Academic_Risk_Probability"
            ].mean() * 100
            if not support_students.empty
            else 0
        )

        support3.metric(
            "Average Risk Probability",
            f"{avg_support_risk:.1f}%"
        )

        st.write("")

        left, right = st.columns(
            [1, 1.25],
            gap="large"
        )

        # Main concerns
        with left:

            with st.container(border=True):

                st.markdown("#### Why Students Need Support")

                st.caption(
                    "Most common warning indicators in this group"
                )

                factor_counts = {
                    "Attendance": int(
                        (
                            support_students["Attendance"] < 70
                        ).sum()
                    ),
                    "Previous Scores": int(
                        (
                            support_students["Previous_Scores"] < 65
                        ).sum()
                    ),
                    "Low Study Hours": int(
                        support_students[
                            "Key_Factors"
                        ].str.contains(
                            "Low Study Hours",
                            case=False,
                            na=False
                        ).sum()
                    ),
                    "Low Motivation": int(
                        support_students[
                            "Key_Factors"
                        ].str.contains(
                            "Low Motivation",
                            case=False,
                            na=False
                        ).sum()
                    ),
                    "Limited Resources": int(
                        support_students[
                            "Key_Factors"
                        ].str.contains(
                            "Limited Resources",
                            case=False,
                            na=False
                        ).sum()
                    )
                }

                factors_df = pd.DataFrame({
                    "Concern": list(factor_counts.keys()),
                    "Students": list(factor_counts.values())
                })

                factors_df = factors_df.sort_values(
                    "Students",
                    ascending=False
                )

                factor_chart = (
                    alt.Chart(factors_df)
                    .mark_bar(
                        cornerRadiusEnd=5
                    )
                    .encode(
                        x=alt.X(
                            "Students:Q",
                            title="Students"
                        ),
                        y=alt.Y(
                            "Concern:N",
                            sort="-x",
                            title=None
                        ),
                        tooltip=[
                            "Concern:N",
                            "Students:Q"
                        ]
                    )
                    .properties(
                        height=280,
                        background="#FFFFFF"
                    )
                    .configure_view(
                        stroke=None
                    )
                )

                st.altair_chart(
                    factor_chart,
                    use_container_width=True,
                    theme=None
                )

        # Recommended support
        with right:

            with st.container(border=True):

                st.markdown("#### Recommended Support")

                st.caption(
                    "Support currently recommended for these students"
                )

                support_types = support_mix(
                    support_students
                )

                support_chart = support_bar_chart(
                    support_types
                )

                if support_chart is not None:

                    st.altair_chart(
                        support_chart,
                        use_container_width=True,
                        theme=None
                    )

                else:

                    st.info(
                        "No recommended support to display."
                    )

    # =====================================================
    # URGENT
    # =====================================================

    elif selected_segment == "urgent":

        urgent_students = analysed[
            urgent_mask
        ].copy()

        st.markdown("### Urgent Attention")

        st.caption(
            "Students with a predicted academic-risk probability "
            "of 75% or higher."
        )

        urgent1, urgent2, urgent3 = st.columns(3)

        urgent1.metric(
            "Urgent Students",
            f"{len(urgent_students):,}"
        )

        urgent_attendance = int(
            (
                urgent_students["Attendance"] < 70
            ).sum()
        )

        urgent2.metric(
            "Below 70% Attendance",
            f"{urgent_attendance:,}"
        )

        urgent_avg = (
            urgent_students[
                "Academic_Risk_Probability"
            ].mean() * 100
            if not urgent_students.empty
            else 0
        )

        urgent3.metric(
            "Average Risk Probability",
            f"{urgent_avg:.1f}%"
        )

        st.write("")

        with st.container(border=True):

            st.markdown("#### Highest Priority Students")

            st.caption(
                "Urgent students ordered by intervention priority"
            )

            urgent_table = urgent_students.sort_values(
                "Intervention_Utility",
                ascending=False
            )

            render_dashboard_student_table(
                urgent_table,
                max_rows=10
            )

    # =====================================================
    # ON TRACK
    # =====================================================

    elif selected_segment == "on-track":

        on_track_students = analysed[
            on_track_mask
        ].copy()

        st.markdown("### Students On Track")

        st.caption(
            "Students who do not currently meet the intervention criteria."
        )

        track1, track2, track3 = st.columns(3)

        avg_track_attendance = (
            on_track_students["Attendance"].mean()
            if not on_track_students.empty
            else 0
        )

        avg_track_score = (
            on_track_students["Previous_Scores"].mean()
            if not on_track_students.empty
            else 0
        )

        avg_track_study = (
            on_track_students["Hours_Studied"].mean()
            if not on_track_students.empty
            else 0
        )

        track1.metric(
            "Average Attendance",
            f"{avg_track_attendance:.1f}%"
        )

        track2.metric(
            "Average Previous Score",
            f"{avg_track_score:.1f}"
        )

        track3.metric(
            "Average Study Hours",
            f"{avg_track_study:.1f} hrs"
        )

        st.write("")

        left, right = st.columns(2, gap="large")

        with left:

            with st.container(border=True):

                st.markdown("#### Attendance")

                st.caption(
                    "Attendance distribution for students currently on track"
                )

                track_attendance_chart = (
                    alt.Chart(on_track_students)
                    .mark_bar(
                        cornerRadiusTopLeft=4,
                        cornerRadiusTopRight=4
                    )
                    .encode(
                        x=alt.X(
                            "Attendance:Q",
                            bin=alt.Bin(step=5),
                            title="Attendance (%)"
                        ),
                        y=alt.Y(
                            "count():Q",
                            title="Students"
                        )
                    )
                    .properties(
                        height=270,
                        background="#FFFFFF"
                    )
                    .configure_view(
                        stroke=None
                    )
                )

                st.altair_chart(
                    track_attendance_chart,
                    use_container_width=True,
                    theme=None
                )

        with right:

            with st.container(border=True):

                st.markdown("#### Previous Performance")

                st.caption(
                    "Previous-score distribution for students currently on track"
                )

                track_score_chart = (
                    alt.Chart(on_track_students)
                    .mark_bar(
                        cornerRadiusTopLeft=4,
                        cornerRadiusTopRight=4
                    )
                    .encode(
                        x=alt.X(
                            "Previous_Scores:Q",
                            bin=alt.Bin(step=5),
                            title="Previous Score"
                        ),
                        y=alt.Y(
                            "count():Q",
                            title="Students"
                        )
                    )
                    .properties(
                        height=270,
                        background="#FFFFFF"
                    )
                    .configure_view(
                        stroke=None
                    )
                )

                st.altair_chart(
                    track_score_chart,
                    use_container_width=True,
                    theme=None
                )

    # =====================================================
    # STUDENTS NEEDING ATTENTION
    # =====================================================

    st.write("")
    st.markdown("---")

    st.markdown("### Students Needing Attention")

    st.caption(
        "Highest-priority students currently eligible for academic support."
    )

    attention_students = analysed[
        analysed["Eligible_For_Intervention"]
    ].copy()

    attention_students = attention_students.sort_values(
        "Intervention_Utility",
        ascending=False
    )

    # -----------------------------------------------------
    # SIMPLE FILTERS
    # -----------------------------------------------------

    filter1, filter2, filter3 = st.columns(
        [1, 1, 2]
    )

    attention_level = filter1.selectbox(
        "Risk Level",
        [
            "All",
            "Low",
            "Medium",
            "High",
            "Very High"
        ],
        key="attention_risk"
    )

    attendance_filter = filter2.selectbox(
        "Attendance",
        [
            "All",
            "Below 70%",
            "70% or Above"
        ],
        key="attention_attendance"
    )

    attention_search = filter3.text_input(
        "Search Student ID",
        placeholder="e.g. S01001",
        key="attention_search"
    )

    if attention_level != "All":

        attention_students = attention_students[
            attention_students["Risk_Category"]
            == attention_level
        ]

    if attendance_filter == "Below 70%":

        attention_students = attention_students[
            attention_students["Attendance"] < 70
        ]

    elif attendance_filter == "70% or Above":

        attention_students = attention_students[
            attention_students["Attendance"] >= 70
        ]

    if attention_search.strip():

        attention_students = attention_students[
            attention_students[
                "Student_ID"
            ].str.contains(
                attention_search.strip(),
                case=False,
                na=False
            )
        ]

    render_dashboard_student_table(
        attention_students,
        max_rows=10
    )

    st.caption(
        "Showing up to 10 students. "
        "Use the Students page to view the full student list."
    )

    # -----------------------------------------------------
    # RESOURCE ALLOCATION LINK / MESSAGE
    # -----------------------------------------------------

    st.write("")

    with st.container(border=True):

        allocation_left, allocation_right = st.columns(
            [3, 1]
        )

        with allocation_left:

            st.markdown(
                "#### Ready to allocate support?"
            )

            st.caption(
                "Use Resource Allocation to prioritise students "
                "when intervention places and resources are limited."
            )

        with allocation_right:

            st.info(
                "Open **Resource Allocation** from the sidebar."
            )

# =========================================================
# STUDENTS
# =========================================================

elif page == "Students":

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    header_left, header_right = st.columns(
        [4, 1]
    )

    with header_left:
        st.markdown("## Students")
        st.caption(
            "View student records, academic risk information "
            "and add new students to the Early Warning System."
        )

    with header_right:

        if st.button(
            "＋ Add Student",
            type="primary",
            use_container_width=True
        ):
            st.session_state.show_add_student = True


    if "show_add_student" not in st.session_state:
        st.session_state.show_add_student = False


    # =====================================================
    # ADD STUDENT FORM
    # =====================================================

    if st.session_state.show_add_student:

        st.write("")

        with st.container(border=True):

            st.markdown("### Add New Student")

            st.caption(
                "Enter the student's academic and contextual information. "
                "The Early Warning System will calculate their risk automatically."
            )

            with st.form(
                "add_student_form",
                clear_on_submit=False
            ):

                # -----------------------------------------
                # ACADEMIC INFORMATION
                # -----------------------------------------

                st.markdown("#### Academic Information")

                academic1, academic2, academic3 = st.columns(3)

                with academic1:

                    new_hours_studied = st.number_input(
                        "Hours Studied",
                        min_value=0,
                        max_value=100,
                        value=20,
                        step=1
                    )

                with academic2:

                    new_attendance = st.number_input(
                        "Attendance (%)",
                        min_value=0,
                        max_value=100,
                        value=75,
                        step=1
                    )

                with academic3:

                    new_previous_scores = st.number_input(
                        "Previous Score",
                        min_value=0,
                        max_value=100,
                        value=70,
                        step=1
                    )


                academic4, academic5, academic6 = st.columns(3)

                with academic4:

                    new_tutoring_sessions = st.number_input(
                        "Tutoring Sessions",
                        min_value=0,
                        max_value=20,
                        value=0,
                        step=1
                    )

                with academic5:

                    new_sleep_hours = st.number_input(
                        "Sleep Hours",
                        min_value=0.0,
                        max_value=24.0,
                        value=7.0,
                        step=0.5
                    )

                with academic6:

                    new_physical_activity = st.number_input(
                        "Physical Activity",
                        min_value=0,
                        max_value=20,
                        value=3,
                        step=1
                    )


                st.write("")

                # -----------------------------------------
                # ENGAGEMENT AND SUPPORT
                # -----------------------------------------

                st.markdown("#### Engagement and Support")

                support1, support2, support3 = st.columns(3)

                with support1:

                    new_motivation = st.selectbox(
                        "Motivation Level",
                        sorted(
                            raw_data[
                                "Motivation_Level"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )

                with support2:

                    new_parental_involvement = st.selectbox(
                        "Parental Involvement",
                        sorted(
                            raw_data[
                                "Parental_Involvement"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )

                with support3:

                    new_resources = st.selectbox(
                        "Access to Resources",
                        sorted(
                            raw_data[
                                "Access_to_Resources"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )


                support4, support5, support6 = st.columns(3)

                with support4:

                    new_extracurricular = st.selectbox(
                        "Extracurricular Activities",
                        sorted(
                            raw_data[
                                "Extracurricular_Activities"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )

                with support5:

                    new_internet = st.selectbox(
                        "Internet Access",
                        sorted(
                            raw_data[
                                "Internet_Access"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )

                with support6:

                    new_peer_influence = st.selectbox(
                        "Peer Influence",
                        sorted(
                            raw_data[
                                "Peer_Influence"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )


                st.write("")

                # -----------------------------------------
                # BACKGROUND INFORMATION
                # -----------------------------------------

                st.markdown("#### Student Background")

                background1, background2, background3 = st.columns(3)

                with background1:

                    new_family_income = st.selectbox(
                        "Family Income",
                        sorted(
                            raw_data[
                                "Family_Income"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )

                with background2:

                    new_parental_education = st.selectbox(
                        "Parental Education Level",
                        sorted(
                            raw_data[
                                "Parental_Education_Level"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )

                with background3:

                    new_distance = st.selectbox(
                        "Distance from Home",
                        sorted(
                            raw_data[
                                "Distance_from_Home"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )


                background4, background5, background6 = st.columns(3)

                with background4:

                    new_school_type = st.selectbox(
                        "School Type",
                        sorted(
                            raw_data[
                                "School_Type"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )

                with background5:

                    new_teacher_quality = st.selectbox(
                        "Teacher Quality",
                        sorted(
                            raw_data[
                                "Teacher_Quality"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )

                with background6:

                    new_gender = st.selectbox(
                        "Gender",
                        sorted(
                            raw_data[
                                "Gender"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )


                background7, background8 = st.columns(2)

                with background7:

                    new_learning_disabilities = st.selectbox(
                        "Learning Disabilities",
                        sorted(
                            raw_data[
                                "Learning_Disabilities"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    )

                with background8:

                    st.info(
                        "The student ID, risk probability, risk level, "
                        "priority score and recommended support will be "
                        "generated automatically."
                    )


                st.write("")

                # -----------------------------------------
                # FORM BUTTONS
                # -----------------------------------------

                button1, button2, button3 = st.columns(
                    [1, 1, 4]
                )

                with button1:

                    submitted = st.form_submit_button(
                        "Add Student",
                        type="primary",
                        use_container_width=True
                    )

                with button2:

                    cancelled = st.form_submit_button(
                        "Cancel",
                        use_container_width=True
                    )


            # ---------------------------------------------
            # CANCEL
            # ---------------------------------------------

            if cancelled:

                st.session_state.show_add_student = False
                st.rerun()


            # ---------------------------------------------
            # ADD STUDENT
            # ---------------------------------------------

            if submitted:

                new_student = {
                    "Hours_Studied": new_hours_studied,
                    "Attendance": new_attendance,
                    "Parental_Involvement": new_parental_involvement,
                    "Access_to_Resources": new_resources,
                    "Extracurricular_Activities": new_extracurricular,
                    "Sleep_Hours": new_sleep_hours,
                    "Previous_Scores": new_previous_scores,
                    "Motivation_Level": new_motivation,
                    "Internet_Access": new_internet,
                    "Tutoring_Sessions": new_tutoring_sessions,
                    "Family_Income": new_family_income,
                    "Teacher_Quality": new_teacher_quality,
                    "School_Type": new_school_type,
                    "Peer_Influence": new_peer_influence,
                    "Physical_Activity": new_physical_activity,
                    "Learning_Disabilities": new_learning_disabilities,
                    "Parental_Education_Level": new_parental_education,
                    "Distance_from_Home": new_distance,
                    "Gender": new_gender
                }


                # -----------------------------------------
                # TEST STUDENT WITH MODEL FIRST
                # -----------------------------------------

                try:

                    test_student = pd.DataFrame(
                        [new_student]
                    )

                    test_probability = model.predict_proba(
                        test_student[REQUIRED_FEATURES]
                    )[:, 1][0]


                    # -------------------------------------
                    # SAVE TO SESSION
                    # -------------------------------------

                    st.session_state.added_students.append(
                        new_student
                    )


                    new_student_number = (
                        len(raw_data)
                        + len(st.session_state.added_students)
                    )

                    generated_id = (
                        f"S{1000 + new_student_number:05d}"
                    )


                    st.session_state.last_added_student = {
                        "Student_ID": generated_id,
                        "Risk_Probability": float(
                            test_probability
                        )
                    }


                    st.session_state.show_add_student = False

                    st.rerun()


                except Exception as exc:

                    st.error(
                        f"Student could not be added: {exc}"
                    )


    # =====================================================
    # SUCCESS MESSAGE
    # =====================================================

    if "last_added_student" in st.session_state:

        added = st.session_state.last_added_student

        new_result = analysed[
            analysed["Student_ID"]
            == added["Student_ID"]
        ]

        if not new_result.empty:

            new_student_result = new_result.iloc[0]

            st.success(
                f"Student {new_student_result['Student_ID']} "
                f"was added successfully."
            )

            result1, result2, result3, result4 = st.columns(4)

            result1.metric(
                "Student ID",
                new_student_result["Student_ID"]
            )

            result2.metric(
                "Risk Level",
                new_student_result["Risk_Category"]
            )

            result3.metric(
                "Risk Probability",
                f"{new_student_result['Academic_Risk_Probability'] * 100:.1f}%"
            )

            result4.metric(
                "Intervention",
                (
                    "Required"
                    if new_student_result[
                        "Eligible_For_Intervention"
                    ]
                    else "Not Required"
                )
            )

            st.write(
                "**Key Factors:** "
                f"{new_student_result['Key_Factors']}"
            )

            st.write(
                "**Recommended Action:** "
                f"{new_student_result['Recommended_Action']}"
            )

            if st.button(
                "Dismiss",
                key="dismiss_added_student"
            ):

                del st.session_state[
                    "last_added_student"
                ]

                st.rerun()


    # =====================================================
    # STUDENT RECORDS
    # =====================================================

    st.write("")
    st.markdown("### Student Records")

    st.caption(
        f"{len(analysed):,} students currently in the system"
    )


    # -----------------------------------------------------
    # FILTERS
    # -----------------------------------------------------

    c1, c2, c3 = st.columns(
        [2, 1, 1]
    )

    search = c1.text_input(
        "Search Student ID",
        placeholder="e.g. S01001",
        key="students_search"
    )

    risk = c2.selectbox(
        "Risk Level",
        [
            "All",
            "Low",
            "Medium",
            "High",
            "Very High"
        ],
        key="students_risk_filter"
    )

    eligibility = c3.selectbox(
        "Status",
        [
            "All",
            "At Risk",
            "On Track"
        ],
        key="students_status_filter"
    )


    # -----------------------------------------------------
    # FILTER DATA
    # -----------------------------------------------------

    student_data = analysed.copy()

    if search.strip():

        student_data = student_data[
            student_data[
                "Student_ID"
            ].str.contains(
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
            student_data[
                "Eligible_For_Intervention"
            ]
        ]

    elif eligibility == "On Track":

        student_data = student_data[
            ~student_data[
                "Eligible_For_Intervention"
            ]
        ]


    # -----------------------------------------------------
    # TABLE
    # -----------------------------------------------------

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
        "Academic_Risk_Probability": "Risk Probability",
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
            "Risk Probability": st.column_config.ProgressColumn(
                "Risk Probability",
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
    top1, top2, top3 = st.columns([1, 1, 2])

    capacity = top1.number_input(
        "Available Intervention Places",
        min_value=1,
        max_value=len(eligible),
        value=min(10, len(eligible)),
        step=1
    )

    budget = top2.number_input(
        "Resource Budget",
        min_value=1,
        value=30,
        step=1,
        help="Maximum resource units available for student interventions."
    )

    top3.markdown(
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
                int(capacity),
                budget=int(budget)
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
        budget_used = int(optimised["Intervention_Cost"].sum())
        budget_remaining = int(budget) - budget_used
        places_used = len(optimised)

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Students Selected",
            f"{places_used} / {int(capacity)}"
        )

        m2.metric(
            "Budget Used",
            f"{budget_used} / {int(budget)}"
        )

        m3.metric(
            "Budget Remaining",
            budget_remaining
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
            "Intervention_Cost",
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
            "Intervention_Cost": "Resource Cost",
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
