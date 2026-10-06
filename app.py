import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from scipy.optimize import milp, LinearConstraint, Bounds



transparent_icon = Image.new("RGBA", (32, 32), (0, 0, 0, 0))

st.set_page_config(
    page_title="Student Early Warning System",
    page_icon=transparent_icon,
    layout="wide"
)

st.markdown(
    """
    <style>
        [data-testid="stToolbar"] {
            display: none !important;
        }

        [data-testid="stDecoration"] {
            display: none !important;
        }

        [data-testid="stStatusWidget"] {
            display: none !important;
        }

        [data-testid="stHeaderActionElements"] {
            display: none !important;
        }

        [data-testid="stHeadingWithActionElements"] a {
            display: none !important;
        }

        #MainMenu {
            visibility: hidden !important;
        }

        header {
            visibility: hidden !important;
            height: 0 !important;
        }

        footer {
            visibility: hidden !important;
        }

        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_FILE = "academic_risk_model.pkl"
DEFAULT_DATA_FILE = "StudentPerformanceFactors.csv"

CV_RESULTS_FILE = "model_comparison_cross_validation.csv"
HELDOUT_METRICS_FILE = "logistic_regression_heldout_metrics.csv"
STRATEGY_RESULTS_FILE = "intervention_strategy_comparison.csv"
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



@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_default_data():
    data = pd.read_csv(DEFAULT_DATA_FILE)

    if "Exam_Score" in data.columns:
        data = data[data["Exam_Score"] <= 100].copy()

    return data


def validate_input_data(data):
    missing = [col for col in REQUIRED_FEATURES if col not in data.columns]
    return missing


def assign_risk_category(probabilities):
    return pd.cut(
        probabilities,
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Low", "Medium", "High", "Very High"]
    )


def calculate_utility(data, weights=BASE_WEIGHTS):
    return (
        weights["ml_risk"] * data["Academic_Risk_Probability"]
        + weights["attendance_risk"] * data["Attendance_Risk_Numeric"]
        + weights["previous_performance_risk"] * data["Previous_Performance_Risk"]
    )


def analyse_students(data, model):
    analysed = data.copy()

    probabilities = model.predict_proba(analysed[REQUIRED_FEATURES])[:, 1]

    analysed["Academic_Risk_Probability"] = probabilities
    analysed["Risk_Category"] = assign_risk_category(probabilities)

    # Fixed rule required by the project.
    analysed["Attendance_Risk"] = analysed["Attendance"] < 70
    analysed["Attendance_Risk_Numeric"] = analysed["Attendance_Risk"].astype(int)

    analysed["Previous_Performance_Risk"] = (
        1 - (analysed["Previous_Scores"] / 100)
    ).clip(0, 1)

    analysed["Intervention_Utility"] = calculate_utility(analysed)

    analysed["Eligible_For_Intervention"] = (
        (analysed["Academic_Risk_Probability"] >= 0.50)
        | analysed["Attendance_Risk"]
    )

    return analysed


def run_milp_optimisation(eligible_students, capacity):
    optimisation_data = (
        eligible_students
        .reset_index()
        .rename(columns={"index": "Student_Index"})
    )

    if len(optimisation_data) == 0 or capacity == 0:
        return optimisation_data.iloc[0:0].copy()

    utility_values = optimisation_data["Intervention_Utility"].to_numpy()
    n_students = len(optimisation_data)

    # scipy.milp minimises, so utility is negated in order to maximise it.
    objective = -utility_values

    # One binary decision variable per student.
    integrality = np.ones(n_students)

    bounds = Bounds(
        lb=np.zeros(n_students),
        ub=np.ones(n_students)
    )

    # Select exactly the available number of intervention places.
    constraint = LinearConstraint(
        np.ones((1, n_students)),
        lb=capacity,
        ub=capacity
    )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraint
    )

    if not result.success:
        raise RuntimeError(result.message)

    selected_positions = np.where(result.x > 0.5)[0]

    selected = (
        optimisation_data
        .iloc[selected_positions]
        .copy()
        .sort_values("Intervention_Utility", ascending=False)
    )

    return selected


def compare_strategies(eligible_students, capacity, random_trials=1000):
    if len(eligible_students) == 0 or capacity == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    capacity = min(capacity, len(eligible_students))

    # Strategy 1: repeated random allocation.
    rng = np.random.default_rng(42)
    eligible_indices = eligible_students.index.to_numpy()

    random_rows = []

    for trial in range(random_trials):
        selected_indices = rng.choice(
            eligible_indices,
            size=capacity,
            replace=False
        )

        selection = eligible_students.loc[selected_indices]

        random_rows.append({
            "Trial": trial + 1,
            "Total_Utility": selection["Intervention_Utility"].sum(),
            "Average_ML_Risk": selection["Academic_Risk_Probability"].mean(),
            "Attendance_Risk_Students": int(selection["Attendance_Risk"].sum())
        })

    random_results = pd.DataFrame(random_rows)

    # Strategy 2: highest ML risk.
    highest_risk_selection = (
        eligible_students
        .sort_values("Academic_Risk_Probability", ascending=False)
        .head(capacity)
    )

    # Strategy 3: optimised utility allocation.
    optimised_selection = run_milp_optimisation(
        eligible_students,
        capacity
    )

    strategy_comparison = pd.DataFrame([
        {
            "Strategy": "Random Allocation",
            "Total_Utility_Mean": random_results["Total_Utility"].mean(),
            "Utility_Std": random_results["Total_Utility"].std(),
            "Average_ML_Risk": random_results["Average_ML_Risk"].mean()
        },
        {
            "Strategy": "Highest ML Risk",
            "Total_Utility_Mean": highest_risk_selection[
                "Intervention_Utility"
            ].sum(),
            "Utility_Std": 0.0,
            "Average_ML_Risk": highest_risk_selection[
                "Academic_Risk_Probability"
            ].mean()
        },
        {
            "Strategy": "Optimised Allocation",
            "Total_Utility_Mean": optimised_selection[
                "Intervention_Utility"
            ].sum(),
            "Utility_Std": 0.0,
            "Average_ML_Risk": optimised_selection[
                "Academic_Risk_Probability"
            ].mean()
        }
    ])

    return (
        strategy_comparison,
        random_results,
        highest_risk_selection,
        optimised_selection
    )


def dataframe_to_csv_bytes(data):
    return data.to_csv(index=False).encode("utf-8")




try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Could not find '{MODEL_FILE}'. "
        "Run the final saving section of your notebook first and place "
        "the saved model in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()



st.sidebar.title("Student EWS")

page = st.sidebar.radio(
    "Page",
    [
        "Dashboard",
        "Student Risk Analysis",
        "Intervention Prioritisation",
        "Model Performance"
    ],
    label_visibility="collapsed"
)

try:
    source_data = load_default_data()
except FileNotFoundError:
    st.error(
        f"Could not find '{DEFAULT_DATA_FILE}'. "
        "Place it in the same folder as app.py."
    )
    st.stop()

missing_features = validate_input_data(source_data)

if missing_features:
    st.error(
        "The selected CSV is missing required model features:\n\n"
        + ", ".join(missing_features)
    )
    st.stop()

try:
    analysed_data = analyse_students(source_data, model)
except Exception as exc:
    st.error(f"Student analysis failed: {exc}")
    st.stop()



# HEADER


st.title("Student Early Warning System")



# DASHBOARD


if page == "Dashboard":

    st.header("Dashboard")

    total_students = len(analysed_data)
    attendance_risk_count = int(analysed_data["Attendance_Risk"].sum())
    ml_high_risk_count = int(
        (analysed_data["Academic_Risk_Probability"] >= 0.50).sum()
    )
    eligible_count = int(analysed_data["Eligible_For_Intervention"].sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Students",
        f"{total_students:,}"
    )

    col2.metric(
        "Attendance Risk",
        f"{attendance_risk_count:,}",
        help="Students with attendance below 70%."
    )

    col3.metric(
        "ML Risk ≥ 50%",
        f"{ml_high_risk_count:,}",
        help="Students with predicted academic risk probability of at least 50%."
    )

    col4.metric(
        "Eligible for Intervention",
        f"{eligible_count:,}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Risk category distribution")

        category_counts = (
            analysed_data["Risk_Category"]
            .value_counts()
            .reindex(["Low", "Medium", "High", "Very High"])
            .fillna(0)
        )

        st.bar_chart(category_counts)

    with right:
        st.subheader("Attendance risk")

        attendance_counts = (
            analysed_data["Attendance_Risk"]
            .map({False: "No Attendance Risk", True: "Attendance Risk"})
            .value_counts()
        )

        st.bar_chart(attendance_counts)

    st.subheader("Highest-priority students")

    priority_columns = [
        "Attendance",
        "Previous_Scores",
        "Academic_Risk_Probability",
        "Risk_Category",
        "Attendance_Risk",
        "Intervention_Utility"
    ]

    top_students = (
        analysed_data
        .sort_values("Intervention_Utility", ascending=False)
        [priority_columns]
        .head(15)
        .copy()
    )

    top_students["Academic_Risk_Probability"] = (
        top_students["Academic_Risk_Probability"] * 100
    ).round(2)

    top_students["Intervention_Utility"] = (
        top_students["Intervention_Utility"]
    ).round(4)

    st.dataframe(
        top_students,
        use_container_width=True
    )


# STUDENT RISK ANALYSIS


elif page == "Student Risk Analysis":

    st.header("Student Risk Analysis")

    st.info(
        "The 70% attendance flag is calculated independently from the "
        "Logistic Regression prediction."
    )

    filters = st.columns(3)

    risk_filter = filters[0].selectbox(
        "Risk category",
        ["All", "Low", "Medium", "High", "Very High"]
    )

    attendance_filter = filters[1].selectbox(
        "Attendance status",
        ["All", "Attendance Risk", "No Attendance Risk"]
    )

    eligibility_filter = filters[2].selectbox(
        "Intervention eligibility",
        ["All", "Eligible", "Not Eligible"]
    )

    filtered = analysed_data.copy()

    if risk_filter != "All":
        filtered = filtered[
            filtered["Risk_Category"].astype(str) == risk_filter
        ]

    if attendance_filter == "Attendance Risk":
        filtered = filtered[filtered["Attendance_Risk"]]

    elif attendance_filter == "No Attendance Risk":
        filtered = filtered[~filtered["Attendance_Risk"]]

    if eligibility_filter == "Eligible":
        filtered = filtered[filtered["Eligible_For_Intervention"]]

    elif eligibility_filter == "Not Eligible":
        filtered = filtered[~filtered["Eligible_For_Intervention"]]

    display_columns = [
        "Hours_Studied",
        "Attendance",
        "Previous_Scores",
        "Academic_Risk_Probability",
        "Risk_Category",
        "Attendance_Risk",
        "Intervention_Utility",
        "Eligible_For_Intervention"
    ]

    if "Exam_Score" in filtered.columns:
        display_columns.insert(3, "Exam_Score")

    display_data = filtered[display_columns].copy()

    display_data["Academic_Risk_Probability"] = (
        display_data["Academic_Risk_Probability"] * 100
    ).round(2)

    display_data["Intervention_Utility"] = (
        display_data["Intervention_Utility"]
    ).round(4)

    st.write(f"Showing **{len(display_data):,}** students.")

    st.dataframe(
        display_data,
        use_container_width=True,
        height=520
    )




# INTERVENTION PRIORITISATION


elif page == "Intervention Prioritisation":

    st.header("Intervention Prioritisation")

    st.write(
        "Eligible students are those with an ML academic risk probability "
        "of at least 50% **or** attendance below 70%."
    )

    st.write(
        "**Intervention utility:** "
        "60% ML risk + 25% attendance risk + "
        "15% previous performance risk."
    )

    eligible_students = analysed_data[
        analysed_data["Eligible_For_Intervention"]
    ].copy()

    eligible_count = len(eligible_students)

    if eligible_count == 0:
        st.success("No students currently meet the intervention eligibility rules.")
        st.stop()

    max_capacity = max(1, eligible_count)

    default_capacity = min(10, max_capacity)

    capacity = st.number_input(
        "Available intervention places",
        min_value=1,
        max_value=max_capacity,
        value=default_capacity,
        step=1
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Eligible Students",
        f"{eligible_count:,}"
    )

    col2.metric(
        "Available Places",
        f"{capacity:,}"
    )

    if st.button(
        "Run Intervention Optimisation",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Calculating intervention priorities..."):

            (
                comparison,
                random_results,
                highest_risk_selection,
                optimised_selection
            ) = compare_strategies(
                eligible_students,
                int(capacity),
                random_trials=1000
            )

        st.success("Optimisation completed.")

        st.subheader("Strategy comparison")

        comparison_display = comparison.copy()

        comparison_display[
            ["Total_Utility_Mean", "Utility_Std", "Average_ML_Risk"]
        ] = comparison_display[
            ["Total_Utility_Mean", "Utility_Std", "Average_ML_Risk"]
        ].round(4)

        st.dataframe(
            comparison_display,
            use_container_width=True
        )

        chart_data = comparison.set_index("Strategy")[
            "Total_Utility_Mean"
        ]

        st.bar_chart(chart_data)

        st.subheader("Recommended intervention priorities")

        selected_display_columns = [
            "Student_Index",
            "Attendance",
            "Previous_Scores",
            "Academic_Risk_Probability",
            "Risk_Category",
            "Attendance_Risk",
            "Intervention_Utility"
        ]

        selected_display = optimised_selection[
            selected_display_columns
        ].copy()

        selected_display["Academic_Risk_Probability"] = (
            selected_display["Academic_Risk_Probability"] * 100
        ).round(2)

        selected_display["Intervention_Utility"] = (
            selected_display["Intervention_Utility"]
        ).round(4)

        st.dataframe(
            selected_display,
            use_container_width=True
        )

        st.download_button(
            "Download intervention priorities",
            data=dataframe_to_csv_bytes(optimised_selection),
            file_name="optimised_intervention_priorities.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download strategy comparison",
            data=dataframe_to_csv_bytes(comparison),
            file_name="intervention_strategy_comparison.csv",
            mime="text/csv"
        )



# MODEL PERFORMANCE


elif page == "Model Performance":

    st.header("Model Performance")

    st.write(
        "The final selected classifier is **Logistic Regression**. "
        "Model selection was performed using stratified cross validation "
        "on the training set, followed by one evaluation on the held out test set."
    )

    try:
        heldout_metrics = pd.read_csv(HELDOUT_METRICS_FILE)

        if not heldout_metrics.empty:
            row = heldout_metrics.iloc[0]

            cols = st.columns(5)

            cols[0].metric(
                "Accuracy",
                f"{row['Accuracy']:.3f}"
            )

            cols[1].metric(
                "Precision",
                f"{row['Precision']:.3f}"
            )

            cols[2].metric(
                "Recall",
                f"{row['Recall']:.3f}"
            )

            cols[3].metric(
                "F1",
                f"{row['F1']:.3f}"
            )

            cols[4].metric(
                "ROC-AUC",
                f"{row['ROC_AUC']:.3f}"
            )

    except FileNotFoundError:
        st.warning(
            f"'{HELDOUT_METRICS_FILE}' was not found. "
            "Run the notebook saving section to display the held out metrics here."
        )

    st.divider()

    try:
        cv_results = pd.read_csv(CV_RESULTS_FILE)

        st.subheader("Cross-validation model comparison")

        st.dataframe(
            cv_results.round(4),
            use_container_width=True
        )

        if (
            "Model" in cv_results.columns
            and "CV_Recall_Mean" in cv_results.columns
        ):
            recall_chart = (
                cv_results
                .set_index("Model")["CV_Recall_Mean"]
            )
            st.bar_chart(recall_chart)

    except FileNotFoundError:
        st.warning(
            f"'{CV_RESULTS_FILE}' was not found. "
            "Run the notebook saving section to display model comparison results."
        )

    left, right = st.columns(2)

    with left:
        try:
            target_sensitivity = pd.read_csv(TARGET_SENSITIVITY_FILE)

            st.subheader("Target threshold sensitivity")
            st.dataframe(
                target_sensitivity.round(4),
                use_container_width=True
            )

        except FileNotFoundError:
            pass

    with right:
        try:
            weight_sensitivity = pd.read_csv(WEIGHT_SENSITIVITY_FILE)

            st.subheader("Utility-weight sensitivity")
            st.dataframe(
                weight_sensitivity.round(4),
                use_container_width=True
            )

        except FileNotFoundError:
            pass

