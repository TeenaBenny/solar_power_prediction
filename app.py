import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="wide"
)

# ----------------------------------
# 2. LOAD TRAINED MODEL
# ----------------------------------

MODEL_PATH = "solar_model.pkl"

st.title("☀️ Solar Power Generation Prediction")

st.write(
    "Predict daily solar energy generation using "
    "weather conditions and solar panel information."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Run solar_power_prediction.py "
        "first to create solar_model.pkl."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

st.success("Machine learning model loaded successfully!")

# ----------------------------------
# 3. USER INPUTS
# ----------------------------------

st.subheader("🌤️ Enter Environmental Conditions")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=60.0,
        value=30.0,
        step=1.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )

    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        max_value=200.0,
        value=10.0,
        step=1.0
    )

    solar_irradiance = st.number_input(
        "Solar Irradiance (W/m²)",
        min_value=0.0,
        max_value=1500.0,
        value=800.0,
        step=10.0
    )

with col2:
    cloud_cover = st.number_input(
        "Cloud Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

    panel_tilt = st.number_input(
        "Panel Tilt (degrees)",
        min_value=0.0,
        max_value=90.0,
        value=25.0,
        step=1.0
    )

    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        index=4,
        format_func=lambda m: pd.Timestamp(
            year=2025, month=m, day=1
        ).strftime("%B")
    )

    day_of_year = st.number_input(
        "Day of Year (1–366)",
        min_value=1,
        max_value=366,
        value=125,
        step=1
    )

# ----------------------------------
# 4. PREDICTION
# ----------------------------------

if st.button(
    "Predict Solar Power Generation",
    type="primary"
):

    # Create input data only when the button is clicked.
    # Feature names and order must match model training.

    input_data = pd.DataFrame([{
        "Temperature_C": temperature,
        "Humidity_pct": humidity,
        "Wind_Speed_kmh": wind_speed,
        "Solar_Irradiance_Wm2": solar_irradiance,
        "Cloud_Cover_pct": cloud_cover,
        "Panel_Tilt_deg": panel_tilt,
        "Month": month,
        "DayOfYear": day_of_year
    }])

    # Generate prediction
    prediction = float(model.predict(input_data)[0])

    # ----------------------------------
    # 5. DISPLAY PREDICTION RESULT
    # ----------------------------------

    st.divider()
    st.subheader("🔆 Prediction Result")

    st.metric(
        label="Estimated Daily Solar Energy",
        value=f"{prediction:.2f} kWh"
    )

    # Demonstration thresholds only.
    # Adjust these for your actual solar installation.

    LOW_LIMIT = 5
    MODERATE_LIMIT = 15
    HIGH_LIMIT = 25

    if prediction < LOW_LIMIT:
        category = "LOW"

        explanation = (
            "The estimated daily energy output is low. "
            "Check sunlight availability, cloud cover, "
            "shading, and the solar installation."
        )

        st.warning("⚠️ LOW SOLAR ENERGY OUTPUT")

    elif prediction < MODERATE_LIMIT:
        category = "MODERATE"

        explanation = (
            "The estimated daily energy output is moderate. "
            "Compare it with the normal output for your "
            "solar installation."
        )

        st.info("MODERATE SOLAR ENERGY OUTPUT")

    elif prediction <= HIGH_LIMIT:
        category = "HIGH"

        explanation = (
            "The estimated daily energy output is high "
            "relative to the demonstration thresholds."
        )

        st.success("✅ HIGH SOLAR ENERGY OUTPUT")

    else:
        category = "VERY HIGH"

        explanation = (
            "The estimated daily energy output is very high "
            "relative to the demonstration thresholds. "
            "Check whether the result is plausible for "
            "your installation."
        )

        st.success("🌞 VERY HIGH SOLAR ENERGY OUTPUT")

    # ----------------------------------
    # 6. INTERPRETATION
    # ----------------------------------

    st.subheader("📊 Interpretation")

    st.write(f"**Output Category:** {category}")

    st.write(explanation)

    st.caption(
        "LOW, MODERATE, HIGH and VERY HIGH use illustrative "
        "thresholds, not validated performance standards. "
        "Actual categories depend on the solar system capacity."
    )

    st.info(
        "This is a machine learning estimate, not a measured "
        "or guaranteed energy output."
    )

    # ----------------------------------
    # 7. INPUT SUMMARY
    # ----------------------------------

    st.subheader("📋 Input Summary")

    st.dataframe(
        input_data,
        use_container_width=True
    )

# ----------------------------------
# 8. PROJECT INFORMATION
# ----------------------------------

st.divider()

st.subheader("ℹ️ About This Project")

st.write("""
**Objective:** Predict daily solar energy generation
from weather and solar panel conditions.

**Machine Learning Algorithm:** Random Forest Regressor

**Problem Type:** Supervised Learning - Regression

**Target Variable:** Power_Output_kWh

**Evaluation Metrics:** MAE, RMSE and R²
""")

st.caption(
    "Educational project. Prediction quality depends on "
    "the training data and the conditions represented in it."
)

# ----------------------------------
# 9. SOLAR ENERGY HELP / FAQ
# ----------------------------------

st.divider()

st.header("💬 Solar Energy Help")

st.write(
    "Choose a question below to learn more about solar "
    "energy generation."
)

question = st.selectbox(
    "Select a question",
    [
        "Why is solar power output low?",
        "How does cloud cover affect generation?",
        "What is solar irradiance?",
        "How can I improve solar panel efficiency?",
        "Why does predicted output differ from actual output?",
        "Which factor affects solar power output most?"
    ]
)

answers = {
    "Why is solar power output low?":
        "Low solar irradiance, heavy cloud cover, shading, "
        "dirty panels, or equipment problems can reduce "
        "solar energy generation.",

    "How does cloud cover affect generation?":
        "Clouds can reduce the sunlight reaching the panels, "
        "usually lowering electricity generation.",

    "What is solar irradiance?":
        "Solar irradiance is the solar power received per "
        "unit area, commonly measured in watts per square metre (W/m²).",

    "How can I improve solar panel efficiency?":
        "Use suitable panel orientation and tilt, reduce "
        "shading, clean panels when needed, and maintain "
        "the system regularly.",

    "Why does predicted output differ from actual output?":
        "Predictions can differ because of weather changes, "
        "measurement errors, equipment conditions, and "
        "limitations in the training dataset.",

    "Which factor affects solar power output most?":
        "In your trained model, solar irradiance had the "
        "highest feature importance. Feature importance "
        "indicates how useful a feature was to the model; "
        "it does not prove causation."
}

st.info(answers[question])

st.caption(
    "Note: Solar Energy Help is a predefined FAQ, "
    "not a generative AI chatbot."
)
