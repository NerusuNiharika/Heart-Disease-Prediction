import streamlit as st
import pickle
import numpy as np

# PAGE SETTINGS

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# DARK UI DESIGN

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}

h1 {
    color: #38bdf8;
    text-align: center;
    font-size: 50px;
}

h3 {
    color: #cbd5e1;
    text-align: center;
}

.main-card {
    background-color: #1e293b;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
}

.stButton>button {
    background-color: #0ea5e9;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #0284c7;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# LOAD MODEL AND SCALER

model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# TITLE

st.markdown(
    "<h1>❤️ Heart Disease Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3>Machine Learning Based Health Risk Predictor</h3>",
    unsafe_allow_html=True
)

st.write("")
st.write("")

# MAIN CARD

with st.container():

    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # LEFT COLUMN

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

        sex = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        cigarettes = st.number_input(
            "Cigarettes Per Day",
            min_value=0,
            max_value=100,
            value=0
        )

    # RIGHT COLUMN

    with col2:

        cholesterol = st.number_input(
            "Total Cholesterol",
            min_value=50,
            max_value=700,
            value=200
        )

        systolic_bp = st.number_input(
            "Systolic Blood Pressure",
            min_value=50,
            max_value=300,
            value=120
        )

        glucose = st.number_input(
            "Glucose Level",
            min_value=40,
            max_value=500,
            value=80
        )

    # CONVERT GENDER

    sex_value = 1 if sex == "Male" else 0

    # PREDICT BUTTON

    if st.button("Predict Heart Disease Risk"):

        # INPUT ARRAY

        input_data = np.array([[
            age,
            sex_value,
            cigarettes,
            cholesterol,
            systolic_bp,
            glucose
        ]])

        # SCALE INPUT

        input_data_scaled = scaler.transform(input_data)

        # PREDICT

        prediction = model.predict(input_data_scaled)

        # PROBABILITY

        probability = model.predict_proba(input_data_scaled)

        disease_probability = probability[0][1] * 100

        # RESULT

        if prediction[0] == 1:

            st.markdown(
                f'''
                <div class="result-box"
                style="background-color:#7f1d1d;color:#fecaca;">
                High Risk of Heart Disease<br><br>
                Risk Probability: {disease_probability:.2f}%
                </div>
                ''',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f'''
                <div class="result-box"
                style="background-color:#052e16;color:#bbf7d0;">
                Low Risk of Heart Disease<br><br>
                Risk Probability: {disease_probability:.2f}%
                </div>
                ''',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)