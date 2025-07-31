import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    """Load the pre-trained Gradient Boosting model."""
    try:
        model = joblib.load("gb_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure it's in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- 3. CUSTOM STYLING (CSS) ---
def apply_custom_styling():
    """Applies custom CSS for aesthetics."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        .stApp {
            background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("https://s7d1.scene7.com/is/image/wbcollab/loan_approved_hero_image:1140x500?qlt=90&fmt=webp&resMode=sharp2");
            background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed;
        }
        [data-testid="stForm"]{
            background-color: rgba(20, 20, 20, 0.7); border-radius: 15px; padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.2); backdrop-filter: blur(8px);
        }
        .result-container { margin-top: 1.5rem; }
        .stButton>button {
            background-color: #4CAF50; color: white; border-radius: 12px; padding: 10px 24px;
            border: none; font-size: 16px; font-weight: bold; width: 100%;
        }
        .stButton>button:hover { background-color: #45a049; }
        .st-emotion-cache-1wmy9hl, .st-emotion-cache-16txtl3, .st-emotion-cache-ue6034, .st-emotion-cache-1y4p8pa, label {
            color: #f0f2f6;
        }
        h1, .stMarkdown p { text-align: center; }
        /* Style for st.metric */
        [data-testid="stMetric"] {
            background-color: rgba(40, 40, 40, 0.5);
            border-radius: 10px;
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SESSION STATE MANAGEMENT ---
def initialize_session_state():
    """Initializes session state for widgets that need default selections."""
    if 'gender' not in st.session_state:
        st.session_state.gender = 'Female'
    if 'education' not in st.session_state:
        st.session_state.education = 'Bachelor'
    if 'ownership' not in st.session_state:
        st.session_state.ownership = 'Rent'
    if 'defaults' not in st.session_state:
        st.session_state.defaults = 'No'
    if 'intent' not in st.session_state:
        st.session_state.intent = 'Debt Consolidation'
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'result_data' not in st.session_state:
        st.session_state.result_data = {}

def reset_form():
    """Callback function to reset inputs and clear the result."""
    st.session_state.prediction_made = False
    st.session_state.result_data = {}
    initialize_session_state()
    validation_placeholder.empty()

# --- 5. MAIN APP LOGIC ---
model = load_model()
if model is None:
    st.stop()

model_columns = model.feature_names_in_
apply_custom_styling()
initialize_session_state()

st.title("🏦 Loan Approval Predictor")
st.markdown("<p style='color: #f0f2f6;'>Enter applicant details to get a real-time loan approval prediction.</p>", unsafe_allow_html=True)

validation_placeholder = st.empty()

# --- Mapping Dictionaries ---
map_gender = {'Female': 'female', 'Male': 'male'}
map_ownership = {'Rent': 'RENT', 'Own': 'OWN', 'Mortgage': 'MORTGAGE', 'Other': 'OTHER'}
map_intent = {
    'Personal': 'PERSONAL', 'Education': 'EDUCATION', 'Medical': 'MEDICAL',
    'Venture': 'VENTURE', 'Home Improvement': 'HOMEIMPROVEMENT', 'Debt Consolidation': 'DEBTCONSOLIDATION'
}
options_education = ['High School', 'Bachelor', 'Master', 'Associate', 'Doctorate']
options_intent = ['Personal', 'Education', 'Medical', 'Venture', 'Home Improvement', 'Debt Consolidation']

# --- 6. INPUT FORM ---
with st.form(key='loan_application_form'):
    col1, col2 = st.columns(2)

    with col1:
        st.header("Personal Information")
        person_income = st.number_input("Annual Income ($)", min_value=1000, step=1000, placeholder="e.g., 60000")
        person_age = st.slider("Age", 18, 80, value=30, help="Applicant's age.")
        person_emp_exp = st.slider("Employment Experience (Years)", 0, 50, value=8, help="Years of professional experience.")
        # CHANGED: Using dropdown (selectbox) for a more compact form
        person_education = st.selectbox("Education Level", options_education, key="education", help="Highest level of education.")
        person_gender_display = st.radio("Gender", ('Female', 'Male'), horizontal=True, key="gender", help="Applicant's gender.")
        
    with col2:
        st.header("Loan & Credit Details")
        loan_amnt = st.number_input("Loan Amount Requested ($)", min_value=500, step=500, placeholder="e.g., 15000")
        loan_percent_income = st.number_input(
            "Loan-to-Income Ratio",
            value=float(f"{(loan_amnt / person_income if person_income and person_income > 0 else 0):.2f}"),
            disabled=True, help="Auto-calculated: (Loan Amount / Annual Income)."
        )
        loan_int_rate = st.slider("Loan Interest Rate (%)", 5.0, 25.0, value=12.5, step=0.01, format="%.2f", help="The proposed interest rate.")
        credit_score = st.slider("Credit Score", 300, 850, value=680, help="Applicant's credit score (e.g., FICO).")
        cb_person_cred_hist_length = st.slider("Credit History Length (Years)", 0, 40, value=6, help="Length of credit history.")
        # CHANGED: Using dropdown (selectbox) for a more compact form
        loan_intent_display = st.selectbox("Loan Purpose", options_intent, key="intent", help="The intended use for the loan.")
        person_home_ownership_display = st.radio("Home Ownership", ('Rent', 'Own', 'Mortgage', 'Other'), horizontal=True, key="ownership")
        previous_loan_defaults_on_file = st.radio("Previous Defaults?", ('No', 'Yes'), horizontal=True, key="defaults")

    submit_button = st.form_submit_button(label='Predict Loan Approval')

result_placeholder = st.empty()

# --- 7. PREDICTION AND VALIDATION LOGIC ---
if submit_button:
    error_messages = []
    if not person_income or not loan_amnt:
        error_messages.append("• Annual Income and Loan Amount are required fields.")
    if person_emp_exp > (person_age - 16):
        error_messages.append("• Employment experience cannot be greater than potential working years (Age - 16).")
    if cb_person_cred_hist_length > (person_age - 16):
        error_messages.append("• Credit history length cannot exceed potential credit-holding years (Age - 16).")
    if person_income and loan_amnt and (loan_amnt > (person_income * 2)):
         error_messages.append("• Loan amount seems excessively high compared to annual income.")

    if error_messages:
        validation_placeholder.error("Please correct the following input errors:\n\n" + "\n".join(error_messages))
    else:
        validation_placeholder.empty()
        st.session_state.prediction_made = True
        
        input_data = {
            'person_age': float(person_age), 'person_income': float(person_income), 'person_emp_exp': person_emp_exp,
            'loan_amnt': float(loan_amnt), 'loan_int_rate': loan_int_rate,
            'loan_percent_income': (loan_amnt / person_income),
            'cb_person_cred_hist_length': float(cb_person_cred_hist_length), 'credit_score': credit_score,
            'person_gender': map_gender[person_gender_display], 'person_education': person_education,
            'person_home_ownership': map_ownership[person_home_ownership_display], 'loan_intent': map_intent[loan_intent_display],
            'previous_loan_defaults_on_file': previous_loan_defaults_on_file
        }
        input_df = pd.DataFrame([input_data])
        input_processed = pd.get_dummies(input_df)
        input_final = input_processed.reindex(columns=model_columns, fill_value=0)
        
        try:
            with st.spinner('Analyzing application...'):
                time.sleep(1)
                prediction = model.predict(input_final)
                prediction_proba = model.predict_proba(input_final)
            st.session_state.result_data = {'prediction': prediction[0], 'probability': prediction_proba[0]}
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- 8. DISPLAY RESULTS ---
if st.session_state.prediction_made:
    with result_placeholder.container():
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        result = st.session_state.result_data
        
        if result:
            prob_rejection = result.get('probability')[0]
            prob_approval = result.get('probability')[1]
            
            if result.get('prediction') == 1:
                st.success(f"**Prediction: Loan Approved** 🎉")
            else:
                st.error(f"**Prediction: Loan Rejected** 😞")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Confidence in Approval", value=f"{prob_approval:.2%}")
            with col2:
                st.metric(label="Confidence in Rejection", value=f"{prob_rejection:.2%}")
            
            st.progress(prob_approval, text="Approval Probability")
            
            st.button("Make another Prediction", on_click=reset_form)
        
        st.markdown('</div>', unsafe_allow_html=True)