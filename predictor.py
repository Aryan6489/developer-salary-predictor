import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

from model_final import prepare_dataframe


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / 'model_3.pkl'
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()
model = data["MODEL"]
label_encoders = data["LABEL_ENCODERS"]
scaler = data["SCALER"]

CURRENCY_OPTIONS = {
    "Indian rupee (INR)": "INR\tIndian rupee",
    "Other / Non-INR currency": "Other"
}

ORG_SIZE_OPTIONS = {
    '2 to 9 employees': '2 to 9 employees',
    '5,000 to 9,999 employees': '5,000 to 9,999 employees',
    '100 to 499 employees': '100 to 499 employees',
    '20 to 99 employees': '20 to 99 employees',
    '1,000 to 4,999 employees': '1,000 to 4,999 employees',
    '10 to 19 employees': '10 to 19 employees',
    '10,000 or more employees': '10,000 or more employees',
    '500 to 999 employees': '500 to 999 employees',
    'Just me - I am a freelancer, sole proprietor, etc.': 'Just me - I am a freelancer, sole proprietor, etc.',
    "I don't know": 'I don�t know'
}

DEV_TYPE_OPTIONS = [
    'Senior Executive (C-Suite, VP, etc.)',
    'Developer, back-end',
    'Developer, front-end',
    'Developer, full-stack',
    'System administrator',
    'Developer, QA or test',
    'Designer',
    'Data scientist or machine learning specialist',
    'Data or business analyst',
    'Security professional',
    'Research & Development role',
    'Developer, mobile',
    'Developer, embedded applications or devices',
    'Developer, desktop or enterprise applications',
    'Engineer, data',
    'Product manager',
    'Cloud infrastructure engineer',
    'Other (please specify):',
    'Developer Experience',
    'Engineering manager',
    'DevOps specialist',
    'Engineer, site reliability',
    'Project manager',
    'Blockchain',
    'Developer, game or graphics',
    'Developer Advocate',
    'Hardware Engineer'
]

REMOTE_WORK_OPTIONS = [
    'Remote',
    'Hybrid (some remote, some in-person)',
    'In-person'
]


def show_predict_page():
    st.title("💼 SalaryScope — Software Developer Salary Prediction")
    st.markdown("### This project predicts salary based on education, experience, developer role, and technology usage.")
    st.markdown(
        "This app uses a survey-trained machine learning model to estimate annual developer compensation. "
        "It considers education, experience, organization size, remote work, and skills to generate a salary estimate."
    )
    st.markdown("---")

    st.write("Please provide the following information:")
    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox("Age Group", [
            "18-24 years old",
            "25-34 years old",
            "35-44 years old",
            "45-54 years old",
            "55-64 years old",
            "65 years or older",
            "Prefer not to say"
        ], help="Select your age group", key="age_select")

        dev_type = st.selectbox("Developer Type", DEV_TYPE_OPTIONS, help="What type of developer are you?", key="dev_type_select")

        orgsize = st.selectbox("Organization Size", list(ORG_SIZE_OPTIONS.keys()), help="Size of your organization", key="orgsize_select")

        aiselect = st.selectbox("AI Tools Usage", ['Yes', "No, and I don't plan to", 'No, but I plan to soon'], help="Do you use AI tools in development?", key="ai_select")

    with col2:
        currency = st.selectbox("Currency", list(CURRENCY_OPTIONS.keys()), help="Your preferred currency", key="currency_select")

        education_level = st.selectbox("Education Level", [
            "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
            "Some college/university study without earning a degree",
            "Master's degree (M.A., M.S., M.Eng., MBA, etc.)",
            "Primary/elementary school",
            "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
            "Associate degree (A.A., A.S., etc.)",
            "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
            "Something else"
        ], help="Your highest education level", key="edu_select")

        experience_input = st.slider("Years of Professional Experience", 0, 50, 3, help="Total years of professional work experience", key="exp_slider")
        yearscode_input = st.slider("Years of Coding Experience", 0, 50, 3, help="Years of coding/programming experience", key="yearscode_slider")
        yearscodepro_input = st.slider("Years of Professional Coding Experience", 0, 50, 3, help="Years of professional coding experience", key="yearscodepro_slider")

    remotework_input = st.selectbox("Work Arrangement", REMOTE_WORK_OPTIONS, help="Your current work setup", key="remote_select")

    st.markdown("### Skills and Technologies")
    col3, col4 = st.columns(2)

    with col3:
        databases_input = st.text_input("Databases Worked With", "", help="List databases separated by semicolons (e.g., MySQL;PostgreSQL)", key="databases_input")

    with col4:
        languages_input = st.text_input("Programming Languages Worked With", "", help="List languages separated by semicolons (e.g., Python;JavaScript)", key="languages_input")

    learning_sources_input = st.text_input("Learning Sources", "", help="Sources you used to learn coding, separated by semicolons", key="learning_input")

    st.markdown("---")
    predict_button = st.button("🔮 Predict Salary", type="primary", use_container_width=True, key="predict_btn")

    if predict_button:
        try:
            with st.spinner("🔄 Calculating your estimated salary..."):
                raw_currency = CURRENCY_OPTIONS[currency]
                raw_orgsize = ORG_SIZE_OPTIONS[orgsize]

                # Build dataframe with all user inputs
                input_data = {
                    'Age': [age],
                    'AISelect': [aiselect],
                    'OrgSize': [raw_orgsize],
                    'DevType': [dev_type],
                    'YearsCode': [int(yearscode_input)],
                    'WorkExp': [int(experience_input)],
                    'YearsCodePro': [int(yearscodepro_input)],
                    'RemoteWork': [remotework_input],
                    'Currency': [raw_currency],
                    'EdLevel': [education_level],
                    'LanguageHaveWorkedWith': [languages_input],
                    'DatabaseHaveWorkedWith': [databases_input],
                    'LearnCode': [learning_sources_input]
                }

                # Create fresh dataframe
                df = pd.DataFrame(input_data)
                
                # Calculate salary prediction
                salary = prepare_dataframe(df, model, label_encoders, scaler)

                st.markdown("---")
                st.markdown(f"<h2 style='text-align: center; color: #28a745;'>💰 Estimated Annual Salary</h2>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; color: #007bff; font-size: 48px;'>${salary[0]:,.2f}</h1>", unsafe_allow_html=True)
                
                # Display breakdown info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Experience Level", f"{experience_input} years")
                with col2:
                    st.metric("Developer Type", dev_type[:25] + "..." if len(dev_type) > 25 else dev_type)
                with col3:
                    st.metric("Currency", currency.split("(")[1].rstrip(")") if "(" in currency else currency)
                
                st.markdown("---")
                st.info("💡 This is an estimate based on survey data and machine learning models. Actual salaries may vary based on location, company, negotiation skills, and other factors.")
        except Exception as e:
            st.error(f"❌ Error calculating salary: {str(e)}")
            st.info("Please check your inputs and try again.")

