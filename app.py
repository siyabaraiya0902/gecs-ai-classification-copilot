import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="GECS AI Copilot", layout="wide")

st.title("Morningstar GECS AI Classification Copilot")

st.write(
    "This prototype sends company and segment text to the Classification API "
    "and returns an industry prediction, subindustry prediction, confidence score, "
    "routing decision, evidence card, and analyst feedback capture."
)

API_URL = "http://127.0.0.1:8000"

# Store result so Streamlit does not forget after button clicks
if "result" not in st.session_state:
    st.session_state.result = None

st.subheader("Input Company and Segment Text")

long_profile = st.text_area(
    "Company Long Profile",
    value="The Company operates as a regional bank providing commercial banking, retail deposits, mortgage lending, credit cards, and small business loans.",
    height=120
)

segment_name = st.text_input(
    "Segment Name",
    value="Regional Banking"
)

segment_description = st.text_area(
    "Segment Description",
    value="Deposit accounts, commercial loans, mortgage banking, credit cards, consumer lending, and branch banking services.",
    height=100
)

if st.button("Classify with API", key="classify_button"):
    payload = {
        "long_profile": long_profile,
        "segment_name": segment_name,
        "segment_description": segment_description
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)

        if response.status_code == 200:
            st.session_state.result = response.json()
            st.success("Prediction received from API")
        else:
            st.error("API returned an error. Check your FastAPI terminal.")
            st.write(response.text)

    except Exception as e:
        st.error("Could not connect to API. Make sure FastAPI is still running.")
        st.write(e)

# Show prediction if we have one
if st.session_state.result is not None:
    result = st.session_state.result

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns([1.4, 1.8, 1])

    with col1:
        st.markdown("**Predicted Industry**")
        st.markdown(f"### {result['predicted_industry']}")

    with col2:
        st.markdown("**Predicted Subindustry**")
        st.markdown(f"### {result['predicted_subindustry']}")

    with col3:
        st.markdown("**Confidence**")
        st.markdown(f"### {result['confidence'] * 100:.1f}%")

    st.subheader("Routing Decision")
    st.info(result["routing"])

    st.subheader("AI Evidence Card")
    st.write("**Reason:**", result["evidence_card"]["reason"])

    if "model_used" in result["evidence_card"]:
        st.write("**Model Used:**", result["evidence_card"]["model_used"])

    st.write("**Top Alternatives:**")

    for alt in result["evidence_card"]["top_alternatives"]:
        if isinstance(alt, dict):
            label = alt.get("label", "Unknown")
            confidence = alt.get("confidence", 0)
            st.write(f"- {label} ({confidence * 100:.1f}%)")
        else:
            st.write("- " + str(alt))

    st.subheader("Analyst Feedback")

    decision = st.radio(
        "What should the analyst do?",
        ["Approve prediction", "Correct prediction", "Escalate for manual review"],
        key="analyst_decision_radio"
    )

    notes = st.text_area("Analyst Notes", key="analyst_notes_box")

    if st.button("Submit Analyst Feedback", key="submit_feedback_button"):
        feedback_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "long_profile": long_profile,
            "segment_name": segment_name,
            "segment_description": segment_description,
            "predicted_industry": result["predicted_industry"],
            "predicted_subindustry": result["predicted_subindustry"],
            "confidence": result["confidence"],
            "routing": result["routing"],
            "analyst_decision": decision,
            "analyst_notes": notes
        }

        feedback_file = os.path.join(os.getcwd(), "feedback_log.csv")

        try:
            existing_feedback = pd.read_csv(feedback_file)
            updated_feedback = pd.concat(
                [existing_feedback, pd.DataFrame([feedback_row])],
                ignore_index=True
            )
        except FileNotFoundError:
            updated_feedback = pd.DataFrame([feedback_row])

        updated_feedback.to_csv(feedback_file, index=False)

        st.success(f"Feedback saved here: {feedback_file}")