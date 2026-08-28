import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Smart Recruitment Assistant",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_model():

    model_data = joblib.load(
        "smart_recruitment_xgb_grid_rfe.pkl"
    )

    return model_data


model_data = load_model()

model = model_data["model"]
selected_features = model_data["selected_features"]
threshold = model_data["threshold"]

experience_median = model_data["experience_median"]
last_new_job_median = model_data["last_new_job_median"]


# ============================================================
# Preprocessing function
# ============================================================

def preprocess_candidates(data):

    candidate = data.copy()

    # Keep ID separately
    enrollee_ids = candidate["enrollee_id"].copy()

    # Drop columns not used by the model
    candidate.drop(
        columns=["enrollee_id", "city"],
        inplace=True,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    candidate["experience"] = candidate["experience"].replace({
        ">20": "21",
        "<1": "0"
    })

    candidate["experience"] = pd.to_numeric(
        candidate["experience"],
        errors="coerce"
    )

    candidate["experience"] = candidate["experience"].fillna(
        experience_median
    )

    # --------------------------------------------------------
    # Last new job
    # --------------------------------------------------------

    candidate["last_new_job"] = candidate["last_new_job"].replace({
        ">4": "5",
        "never": "0"
    })

    candidate["last_new_job"] = pd.to_numeric(
        candidate["last_new_job"],
        errors="coerce"
    )

    candidate["last_new_job"] = candidate["last_new_job"].fillna(
        last_new_job_median
    )

    # --------------------------------------------------------
    # Company size
    # --------------------------------------------------------

    candidate["company_size"] = candidate["company_size"].replace({
        "10/49": "10-49"
    })

    # --------------------------------------------------------
    # Missing categorical values
    # --------------------------------------------------------

    cat_columns = [
        "company_type",
        "company_size",
        "gender",
        "major_discipline",
        "education_level",
        "enrolled_university"
    ]

    for col in cat_columns:

        candidate[col] = candidate[col].fillna("unknown")

    # --------------------------------------------------------
    # Relevant experience
    # --------------------------------------------------------

    candidate["relevent_experience"] = candidate[
        "relevent_experience"
    ].map({
        "Has relevent experience": 1,
        "No relevent experience": 0
    })

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    education_map = {
        "Primary School": 0,
        "High School": 1,
        "Graduate": 2,
        "Masters": 3,
        "Phd": 4,
        "unknown": -1
    }

    candidate["education_level"] = candidate[
        "education_level"
    ].map(education_map)

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    ohe_cols = [
        "company_type",
        "company_size",
        "gender",
        "major_discipline",
        "enrolled_university"
    ]

    candidate = pd.get_dummies(
        candidate,
        columns=ohe_cols,
        dtype=int
    )

    # Rename columns
    candidate.rename(
        columns={
            "company_size_<10": "company_size_under_10",
            "company_size_10000+": "company_size_morethan_10000"
        },
        inplace=True
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    candidate["career_stability"] = (
        candidate["last_new_job"] /
        (candidate["experience"] + 1)
    )

    candidate["training_intensity"] = (
        candidate["training_hours"] /
        (candidate["experience"] + 1)
    )

    # --------------------------------------------------------
    # Make columns exactly match training features
    # --------------------------------------------------------

    candidate = candidate.reindex(
        columns=selected_features,
        fill_value=0
    )

    return enrollee_ids, candidate


# ============================================================
# Prediction function
# ============================================================

def predict_candidates(data):

    enrollee_ids, processed_data = preprocess_candidates(data)

    # Get probabilities for BOTH classes
    probabilities = model.predict_proba(processed_data)

    # Class 0 = Not looking for a new job
    probability_not_looking = probabilities[:, 0]

    # Class 1 = Looking for a new job
    probability_looking = probabilities[:, 1]

    # Classification based on your selected threshold
    predictions = (
        probability_looking >= threshold
    ).astype(int)

    results = pd.DataFrame({
        "enrollee_id": enrollee_ids.to_numpy(),
        "probability_not_looking": probability_not_looking,
        "probability_looking": probability_looking,
        "prediction": predictions
    })

    results["result"] = results["prediction"].map({
        0: "Likely not looking for a new job",
        1: "Likely looking for a new job"
    })

    return results


# ============================================================
# Title
# ============================================================

st.title("💼 Smart Recruitment Assistant")

st.write(
    "Predict the probability of a candidate to look for a new job or will work for the company."
)

st.divider()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Model Information")

st.sidebar.write(
    f"Classification threshold: **{threshold:.2f}**"
)

st.sidebar.write(
    "Class 1 = Looking for a new job"
)

st.sidebar.write(
    "Class 0 = Not looking for a new job"
)


# ============================================================
# Tabs
# ============================================================

tab1, tab2 = st.tabs([
    "👤 Single Candidate",
    "🏆 Top 10 Candidates"
])


# ============================================================
# SINGLE CANDIDATE
# ============================================================

with tab1:

    st.header("Candidate Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:

        enrollee_id = st.number_input(
            "Enrollee ID",
            min_value=0,
            value=1000
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other",
                "unknown"
            ]
        )

        relevent_experience = st.selectbox(
            "Relevant Experience",
            [
                "Has relevent experience",
                "No relevent experience"
            ]
        )

        education_level = st.selectbox(
            "Education Level",
            [
                "Primary School",
                "High School",
                "Graduate",
                "Masters",
                "Phd",
                "unknown"
            ]
        )

    with col2:

        experience = st.selectbox(
            "Experience",
            [
                "<1",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                ">20"
            ]
        )

        last_new_job = st.selectbox(
            "Years Since Last Job",
            [
                "never",
                "1",
                "2",
                "3",
                "4",
                ">4"
            ]
        )

        training_hours = st.number_input(
            "Training Hours",
            min_value=0,
            max_value=500,
            value=50
        )

        city_development_index = st.number_input(
            "City Development Index",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01
        )

    with col3:

        company_size = st.selectbox(
            "Company Size",
            [
                "<10",
                "10-49",
                "50-99",
                "100-500",
                "500-999",
                "1000-4999",
                "5000-9999",
                "10000+",
                "unknown"
            ]
        )

        company_type = st.selectbox(
            "Company Type",
            [
                "Pvt Ltd",
                "Funded Startup",
                "Early Stage Startup",
                "Other",
                "Public Sector",
                "NGO",
                "unknown"
            ]
        )

        major_discipline = st.selectbox(
            "Major Discipline",
            [
                "STEM",
                "Business Degree",
                "Arts",
                "Humanities",
                "No Major",
                "Other",
                "unknown"
            ]
        )

        enrolled_university = st.selectbox(
            "University Enrollment",
            [
                "no_enrollment",
                "Part time course",
                "Full time course",
                "unknown"
            ]
        )

    if st.button(
        "Predict Candidate",
        type="primary"
    ):

        candidate_input = pd.DataFrame([{
            "enrollee_id": enrollee_id,
            "gender": gender,
            "relevent_experience": relevent_experience,
            "education_level": education_level,
            "experience": experience,
            "last_new_job": last_new_job,
            "training_hours": training_hours,
            "city_development_index": city_development_index,
            "company_size": company_size,
            "company_type": company_type,
            "major_discipline": major_discipline,
            "enrolled_university": enrolled_university
        }])

        results = predict_candidates(
            candidate_input
        )

        probability_looking = results.iloc[0][
            "probability_looking"
        ]

        probability_not_looking = results.iloc[0][
            "probability_not_looking"
        ]

        prediction = results.iloc[0]["prediction"]

        result = results.iloc[0]["result"]

        st.divider()

        st.subheader("Prediction")

        col1, col2 = st.columns(2)
        
        with col1:
        
            st.metric(
                "Not looking for a new job",
                f"{probability_not_looking * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Looking for a new job",
                f"{probability_looking * 100:.2f}%"
            )

        if prediction == 1:

            st.success(
                f"🟢 {result}"
            )

        else:

            st.info(
                f"🔵 {result}"
            )


# ============================================================
# TOP 10 CANDIDATES
# ============================================================

with tab2:

    st.header("🏆 Top 10 Candidates")

    st.write(
        "Upload candidate data to rank candidates according "
        "to their probability of looking for a new job."
    )

    uploaded_file = st.file_uploader(
        "Upload candidate CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        candidates = pd.read_csv(
            uploaded_file
        )

        st.subheader("Uploaded Candidates")

        st.dataframe(
            candidates.head(),
            use_container_width=True
        )

        try:

            results = predict_candidates(
                candidates
            )

            # Rank by probability of looking
            results = results.sort_values(
                by="probability_not_looking",
                ascending=False
            ).reset_index(drop=True)

            results["rank"] = (
                results.index + 1
            )

            top_10 = results.head(10)

            st.subheader(
                "Top 10 Candidates"
            )

            display_results = top_10[
                [
                    "rank",
                    "enrollee_id",
                    "probability_looking",
                    "probability_not_looking",
                    "result"
                ]
            ].copy()

            display_results[
                "probability_not_looking"
            ] = (
                display_results[
                    "probability_looking"
                ] * 100
            ).round(2)

            display_results[
                "probability_looking"
            ] = (
                display_results[
                    "probability_not_looking"
                ] * 100
            ).round(2)

            display_results.rename(
                columns={
                    "rank": "Rank",
                    "enrollee_id": "Candidate ID",
                    "probability_looking":
                        "Looking Probability (%)",
                    "probability_not_looking":
                        "Not Looking Probability (%)",
                    "result": "Prediction"
                },
                inplace=True
            )

            st.dataframe(
                display_results,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Download results
            # ------------------------------------------------

            csv = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download All Predictions",
                data=csv,
                file_name="candidate_results.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Error processing candidates: {e}"
            )