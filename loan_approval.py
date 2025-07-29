import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time

# --- Page Configuration ---
# This must be the first Streamlit command in your script.
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
)

# --- Function to Load Model ---
# Using st.cache_resource ensures the model is loaded only once.
@st.cache_resource
def load_model():
    try:
        model = joblib.load("gb_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'gb_model.pkl' not found. Please ensure it's in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- Function for Custom Styling ---
def apply_custom_styling():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        /* Target the body for the background image to avoid z-index issues with dropdowns */
        .stApp {
            background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("https://s7d1.scene7.com/is/image/wbcollab/loan_approved_hero_image:1140x500?qlt=90&fmt=webp&resMode=sharp2");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
            
        /* form container with translucent overlay */
        [data-testid="stForm"] {
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 15px;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(5px);
        }
        
        .st-emotion-cache-1wmy9hl e1f1d6gn1{
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 15px;
            padding: 2rem;    
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(5px);
        }               
        .stButton>button {
            background-color: #4CAF50; color: white; border-radius: 12px;
            padding: 10px 24px; border: none; font-size: 16px;
            font-weight: bold; width: 100%;
        }
        .stButton>button:hover { background-color: #45a049; }
    </style>
    """, unsafe_allow_html=True)

# --- Function to Initialize or Reset Session State ---
def initialize_session_state():
    defaults = {
        'age': 25, 'gender': 'Female', 'education': 'High School', 'income': 50000,
        'emp_exp': 5, 'ownership': 'Rent', 'defaults': 'No', 'loan_amount': 10000,
        'intent': 'Personal', 'interest_rate': 11.5, 'credit_length': 4, 'credit_score': 650
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'result_data' not in st.session_state:
        st.session_state.result_data = {}

def reset_form():
    """Callback function to reset the form state and clear inputs."""
    st.session_state.prediction_made = False
    st.session_state.result_data = {}
    # Reset all widget values to their defaults
    st.session_state.age = 25
    st.session_state.gender = 'Female'
    st.session_state.education = 'High School'
    st.session_state.income = 50000
    st.session_state.emp_exp = 5
    st.session_state.ownership = 'Rent'
    st.session_state.defaults = 'No'
    st.session_state.loan_amount = 10000
    st.session_state.intent = 'Personal'
    st.session_state.interest_rate = 11.5
    st.session_state.credit_length = 4
    st.session_state.credit_score = 650
    validation_placeholder.empty()

# --- Main App Logic ---
model = load_model()
if model is None:
    st.stop()

model_columns = model.feature_names_in_
apply_custom_styling()
initialize_session_state()

# --- App Title and Description ---
st.title("🏦 Loan Approval Predictor")
st.markdown("<p style='text-align: center; color: #f0f2f6;'>Enter the applicant's details below to get a real-time loan approval prediction.</p>", unsafe_allow_html=True)

# --- Placeholder for Validation Messages ---
validation_placeholder = st.empty()

# --- Define User-Friendly Options and Mappings ---
options_gender = ['Female', 'Male']
options_education = ['High School', 'Bachelor', 'Master', 'Associate', 'Doctorate']
options_ownership = ['Rent', 'Own', 'Mortgage', 'Other']
options_intent = ['Personal', 'Education', 'Medical', 'Venture', 'Home Improvement', 'Debt Consolidation']

map_gender = {'Female': 'female', 'Male': 'male'}
map_ownership = {'Rent': 'RENT', 'Own': 'OWN', 'Mortgage': 'MORTGAGE', 'Other': 'OTHER'}
map_intent = {
    'Personal': 'PERSONAL', 'Education': 'EDUCATION', 'Medical': 'MEDICAL',
    'Venture': 'VENTURE', 'Home Improvement': 'HOMEIMPROVEMENT', 'Debt Consolidation': 'DEBTCONSOLIDATION'
}

# --- Create Input Fields for User Data ---
with st.form(key='loan_application_form'):
    st.header("Applicant's Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Age", 18, 100, key="age", help="Applicant's age.")
        person_gender_display = st.radio("Gender", options_gender, horizontal=True, key="gender", help="Applicant's gender.")
        person_education = st.selectbox("Education Level", options_education, key="education", help="Highest level of education.")
        person_income = st.number_input("Annual Income ($)", 0, None, step=1000, key="income", help="Applicant's total annual income.")
    with col2:
        person_emp_exp = st.number_input("Employment Experience (Years)", 0, 50, key="emp_exp", help="Years of professional experience.")
        person_home_ownership_display = st.selectbox("Home Ownership", options_ownership, key="ownership", help="Status of home ownership.")
        previous_loan_defaults_on_file = st.radio("Previous Defaults?", ('No', 'Yes'), horizontal=True, key="defaults", help="Has the applicant ever defaulted?")

    st.header("Loan Details")
    col3, col4 = st.columns(2)
    with col3:
        loan_amnt = st.number_input("Loan Amount Requested ($)", 500, None, step=500, key="loan_amount", help="The total amount requested.")
        loan_intent_display = st.selectbox("Loan Purpose", options_intent, key="intent", help="The intended use for the loan.")
        loan_int_rate = st.number_input("Loan Interest Rate (%)", 0.0, 30.0, format="%.2f", key="interest_rate", help="The proposed interest rate.")
    with col4:
        loan_percent_income = st.number_input(
            "Loan to Income Ratio", 0.0, 1.0, value=float(f"{(loan_amnt / person_income if person_income > 0 else 0):.2f}"),
            format="%.2f", disabled=True, help="Auto-calculated: (Loan Amount / Annual Income)."
        )
        cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", 0, 40, key="credit_length", help="Length of credit history.")
        credit_score = st.number_input("Credit Score", 300, 850, key="credit_score", help="Applicant's credit score.")
    
    submit_button = st.form_submit_button(label='Predict Loan Approval')

# --- Anchor for scrolling ---
st.markdown('<a id="result-anchor"></a>', unsafe_allow_html=True)
result_placeholder = st.empty()


# --- Prediction and Validation Logic ---
if submit_button:
    error_messages = []
    if person_emp_exp > (person_age - 16):
        error_messages.append("• Employment experience cannot be greater than potential working years (Age - 16).")
    if cb_person_cred_hist_length > (person_age - 16):
        error_messages.append("• Credit history length cannot be greater than potential credit-holding years (Age - 16).")

    if error_messages:
        validation_placeholder.error("Please correct the following input errors:\n\n" + "\n".join(error_messages))
    else:
        validation_placeholder.empty()
        st.session_state.prediction_made = True
        
        input_data = {
            'person_age': float(person_age), 'person_income': float(person_income), 'person_emp_exp': person_emp_exp,
            'loan_amnt': float(loan_amnt), 'loan_int_rate': loan_int_rate,
            'loan_percent_income': (loan_amnt / person_income) if person_income > 0 else 0,
            'cb_person_cred_hist_length': float(cb_person_cred_hist_length), 'credit_score': credit_score,
            'person_gender': map_gender[person_gender_display], 'person_education': person_education,
            'person_home_ownership': map_ownership[person_home_ownership_display], 'loan_intent': map_intent[loan_intent_display],
            'previous_loan_defaults_on_file': previous_loan_defaults_on_file
        }
        input_df = pd.DataFrame([input_data])
        input_processed = pd.get_dummies(input_df)
        input_final = input_processed.reindex(columns=model_columns, fill_value=0)
        
        try:
            prediction = model.predict(input_final)
            prediction_proba = model.predict_proba(input_final)
            st.session_state.result_data = {'prediction': prediction[0], 'probability': prediction_proba[0]}
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- Display Results ---
if st.session_state.prediction_made:
    with result_placeholder.container():
        st.subheader("Prediction Result")
        result = st.session_state.result_data
        if result.get('prediction') == 0:
            st.error(f"**Prediction: Loan Rejected** 😞")
            st.write(f"The model predicts a **{result.get('probability')[0]:.2%}** probability of rejection.")
        else:
            st.success(f"**Prediction: Loan Approved** 🎉")
            st.write(f"The model predicts a **{result.get('probability')[1]:.2%}** probability of approval.")
        
        st.button("Make a New Prediction", on_click=reset_form)

        st.components.v1.html("""
            <script>
                setTimeout(function() {
                    const anchor = document.getElementById("result-anchor");
                    if (anchor) {
                        anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }, 100);
            </script>
        """, height=0)
