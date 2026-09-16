# ============================================================
# PACS CREDIT RISK ASSESSMENT
# Standalone Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PACS Credit Risk Assessment",
    page_icon="🌾",
    layout="centered"
)


# ============================================================
# 2. MODEL FILES
# ============================================================

MODEL_FILE = "pacs_final_random_forest_model.joblib"
PREPROCESSOR_FILE = "pacs_final_preprocessor.joblib"
THRESHOLD_FILE = "pacs_final_threshold.joblib"
CONFIG_FILE = "pacs_model_configuration.joblib"


# ============================================================
# 3. CHECK MODEL FILES
# ============================================================

required_files = [
    MODEL_FILE,
    PREPROCESSOR_FILE,
    THRESHOLD_FILE,
    CONFIG_FILE
]

missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error("The following model files are missing:")

    for file in missing_files:
        st.write(f"- {file}")

    st.info(
        "Place all four .joblib files in the same folder "
        "as app.py."
    )

    st.stop()


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_FILE
    )

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    threshold = joblib.load(
        THRESHOLD_FILE
    )

    configuration = joblib.load(
        CONFIG_FILE
    )

    return (
        model,
        preprocessor,
        threshold,
        configuration
    )


model, preprocessor, threshold, configuration = load_model()


# ============================================================
# 5. HEADER
# ============================================================

st.title(
    "🌾 PACS Credit Risk Assessment"
)

st.markdown(
    """
    ### Agricultural Credit Risk Prediction

    Enter the farmer's financial, agricultural and repayment
    information below to generate a model-based credit-risk
    assessment.
    """
)

st.divider()


# ============================================================
# 6. FINANCIAL INFORMATION
# ============================================================

st.subheader(
    "1. Financial Information"
)

col1, col2 = st.columns(2)


with col1:

    previous_loan_count = st.number_input(
        "Previous Loan Count",
        min_value=0,
        max_value=50,
        value=2,
        step=1
    )

    annual_farm_income = st.number_input(
        "Annual Farm Income (₹)",
        min_value=0.0,
        value=250000.0,
        step=10000.0
    )

    annual_farm_expenses = st.number_input(
        "Annual Farm Expenses (₹)",
        min_value=0.0,
        value=120000.0,
        step=10000.0
    )

    outstanding_debt = st.number_input(
        "Outstanding Debt (₹)",
        min_value=0.0,
        value=50000.0,
        step=5000.0
    )


with col2:

    requested_loan_amount = st.number_input(
        "Requested Loan Amount (₹)",
        min_value=0.0,
        value=75000.0,
        step=5000.0
    )

    loan_tenure_months = st.selectbox(
        "Loan Tenure (months)",
        options=[
            6,
            12,
            18,
            24,
            36,
            48,
            60
        ],
        index=1
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        options=[
            "crop",
            "livestock_allied",
            "equipment",
            "irrigation",
            "other"
        ]
    )


# ============================================================
# 7. DEBT-TO-INCOME RATIO
# ============================================================

if annual_farm_income > 0:

    debt_to_income = (
        outstanding_debt
        /
        annual_farm_income
    )

else:

    debt_to_income = 0.0


st.caption(
    f"Calculated Debt-to-Income Ratio: "
    f"{debt_to_income:.3f}"
)


# ============================================================
# 8. FARM INFORMATION
# ============================================================

st.subheader(
    "2. Farm Information"
)

col1, col2 = st.columns(2)


with col1:

    cultivated_area = st.number_input(
        "Cultivated Area (acres)",
        min_value=0.0,
        value=2.0,
        step=0.1
    )

    farming_experience_years = st.number_input(
        "Farming Experience (years)",
        min_value=0,
        max_value=80,
        value=10,
        step=1
    )

    crop_type = st.selectbox(
        "Primary Crop",
        options=[
            "Maize",
            "Rice",
            "Wheat",
            "Pulses",
            "Vegetables",
            "Cotton",
            "Other"
        ]
    )


with col2:

    irrigation_available = st.selectbox(
        "Irrigation Available?",
        options=[
            "Yes",
            "No"
        ]
    )


irrigation_binary = (
    1
    if irrigation_available == "Yes"
    else 0
)


# ============================================================
# 9. REPAYMENT HISTORY
# ============================================================

st.subheader(
    "3. Repayment History"
)

repayment_history_available = st.selectbox(
    "Previous Repayment History Available?",
    options=[
        "Yes",
        "No"
    ]
)


if repayment_history_available == "Yes":

    repayment_rate = st.slider(
        "Repayment Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
        format="%.2f"
    )

    avg_delay_days = st.number_input(
        "Average Repayment Delay (days)",
        min_value=0.0,
        max_value=365.0,
        value=10.0,
        step=1.0
    )

    repayment_history_missing = 0
    avg_delay_history_missing = 0

else:

    repayment_rate = np.nan

    avg_delay_days = np.nan

    repayment_history_missing = 1
    avg_delay_history_missing = 1

    st.info(
        "Repayment history is unavailable. "
        "The model will handle these values as missing."
    )


# ============================================================
# 10. AGRICULTURAL RISK
# ============================================================

st.subheader(
    "4. Agricultural Risk"
)

weather_risk_score = st.slider(
    "Weather Risk Score",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.01,
    help=(
        "0 = lower weather risk, "
        "1 = higher weather risk"
    )
)


# ============================================================
# 11. ASSESS BUTTON
# ============================================================

st.divider()

assess_button = st.button(
    "🔍 Assess Credit Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# 12. RUN PREDICTION
# ============================================================

if assess_button:

    try:

        # ====================================================
        # BUILD RAW FARMER DATA
        # ====================================================

        farmer_data = {

            "previous_loan_count":
                previous_loan_count,

            "repayment_rate":
                repayment_rate,

            "avg_delay_days":
                avg_delay_days,

            "outstanding_debt":
                outstanding_debt,

            "cultivated_area":
                cultivated_area,

            "farming_experience_years":
                farming_experience_years,

            "annual_farm_income":
                annual_farm_income,

            "annual_farm_expenses":
                annual_farm_expenses,

            "debt_to_income":
                debt_to_income,

            "requested_loan_amount":
                requested_loan_amount,

            "weather_risk_score":
                weather_risk_score,

            "crop_type":
                crop_type,

            "loan_purpose":
                loan_purpose,

            "loan_tenure_months":
                loan_tenure_months,

            "irrigation_available":
                irrigation_binary,

            "repayment_history_missing":
                repayment_history_missing,

            "avg_delay_history_missing":
                avg_delay_history_missing
        }


        # ====================================================
        # DATAFRAME
        # ====================================================

        farmer_df = pd.DataFrame(
            [farmer_data]
        )


        # ====================================================
        # ENSURE CORRECT FEATURE ORDER
        # ====================================================

        expected_features = (
            configuration["numerical_features"]
            +
            configuration["categorical_features"]
            +
            configuration["binary_features"]
        )

        farmer_df = farmer_df[
            expected_features
        ]


        # ====================================================
        # APPLY SAVED PREPROCESSOR
        # ====================================================

        farmer_processed = (
            preprocessor.transform(
                farmer_df
            )
        )


        # ====================================================
        # BASE RANDOM FOREST PREDICTION
        # ====================================================

        base_probability = float(
            model.predict_proba(
                farmer_processed
            )[0, 1]
        )


        # ====================================================
        # DOMAIN-INFORMED RISK ADJUSTMENT
        # ====================================================
        #
        # The Random Forest remains the base ML model.
        #
        # Because the project dataset is synthetic, this
        # prototype uses a small domain-informed adjustment
        # layer so that major financial-risk factors have a
        # visible and consistent effect.
        #
        # The final value is therefore an:
        #
        #     ADJUSTED DEFAULT-RISK ESTIMATE
        #
        # rather than a pure Random Forest probability.
        #
        # ====================================================

        risk_adjustment = 0.0


        # ====================================================
        # A. DEBT-TO-INCOME RATIO
        # ====================================================

        if debt_to_income <= 0.20:

            risk_adjustment -= 0.04

        elif debt_to_income <= 0.40:

            risk_adjustment += 0.00

        elif debt_to_income <= 0.60:

            risk_adjustment += 0.05

        elif debt_to_income <= 0.80:

            risk_adjustment += 0.12

        elif debt_to_income <= 1.00:

            risk_adjustment += 0.20

        else:

            # Debt exceeds annual farm income
            risk_adjustment += 0.30


        # ====================================================
        # B. REQUESTED LOAN / ANNUAL INCOME
        # ====================================================

        if annual_farm_income > 0:

            loan_income_ratio = (
                requested_loan_amount
                /
                annual_farm_income
            )

        else:

            loan_income_ratio = 1.0


        if loan_income_ratio <= 0.20:

            risk_adjustment -= 0.03

        elif loan_income_ratio <= 0.40:

            risk_adjustment += 0.00

        elif loan_income_ratio <= 0.60:

            risk_adjustment += 0.04

        elif loan_income_ratio <= 0.80:

            risk_adjustment += 0.08

        else:

            risk_adjustment += 0.12


        # ====================================================
        # C. REPAYMENT HISTORY
        # ====================================================

        if repayment_history_missing == 0:

            # -----------------------------------------------
            # Repayment rate
            # -----------------------------------------------

            if repayment_rate >= 0.90:

                risk_adjustment -= 0.08

            elif repayment_rate >= 0.75:

                risk_adjustment += 0.00

            elif repayment_rate >= 0.60:

                risk_adjustment += 0.08

            else:

                risk_adjustment += 0.16


            # -----------------------------------------------
            # Average delay
            # -----------------------------------------------

            if avg_delay_days <= 5:

                risk_adjustment -= 0.04

            elif avg_delay_days <= 15:

                risk_adjustment += 0.00

            elif avg_delay_days <= 30:

                risk_adjustment += 0.06

            else:

                risk_adjustment += 0.12

        else:

            # Missing repayment information introduces
            # uncertainty, but does not automatically mean
            # high risk.

            risk_adjustment += 0.03


        # ====================================================
        # D. OUTSTANDING DEBT
        # ====================================================

        if annual_farm_income > 0:

            outstanding_debt_ratio = (
                outstanding_debt
                /
                annual_farm_income
            )

        else:

            outstanding_debt_ratio = 1.0


        if outstanding_debt_ratio > 1.0:

            risk_adjustment += 0.08

        elif outstanding_debt_ratio > 0.60:

            risk_adjustment += 0.04


        # ====================================================
        # E. WEATHER RISK
        # ====================================================

        if weather_risk_score <= 0.30:

            risk_adjustment -= 0.02

        elif weather_risk_score <= 0.60:

            risk_adjustment += 0.03

        else:

            risk_adjustment += 0.08


        # ====================================================
        # F. IRRIGATION
        # ====================================================

        if irrigation_binary == 0:

            risk_adjustment += 0.04

        else:

            risk_adjustment -= 0.02


        # ====================================================
        # G. LIMIT ADJUSTMENT
        # ====================================================

        risk_adjustment = np.clip(
            risk_adjustment,
            -0.12,
            0.45
        )


        # ====================================================
        # H. FINAL ADJUSTED PROBABILITY
        # ====================================================

        probability = (
            base_probability
            +
            risk_adjustment
        )

        probability = np.clip(
            probability,
            0.01,
            0.99
        )
        # ============================================================
        # CREDIT SCORE
        # ============================================================

        credit_score = round(
            (1 - probability) * 100
        )

        credit_score = int(
            np.clip(
                credit_score,
                1,
                99
            )
        )


        # ====================================================
        # I. MODEL DECISION
        # ====================================================

        model_prediction = int(
            probability >= threshold
        )


        # ====================================================
        # J. RISK CATEGORY
        # ====================================================

        if probability < 0.30:

            risk_category = "Low"

        elif probability < 0.60:

            risk_category = "Medium"

        else:

            risk_category = "High"


        # ====================================================
        # K. MODEL SIGNAL
        # ====================================================

        if model_prediction == 1:

            model_signal = (
                "Higher default-risk signal"
            )

        else:

            model_signal = (
                "Lower default-risk signal"
            )


        # ====================================================
        # 13. DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "Credit Risk Assessment"
        )


        # ====================================================
        # PROBABILITY
        # ====================================================

        st.metric(
            "Adjusted Default-Risk Estimate",
            f"{probability * 100:.2f}%"
        )
        st.metric(
            "Credit Score",
            f"{credit_score} / 100"
        )


        # ====================================================
        # RISK CATEGORY
        # ====================================================

        if risk_category == "Low":

            st.success(
                f"Risk Category: {risk_category}"
            )

        elif risk_category == "Medium":

            st.warning(
                f"Risk Category: {risk_category}"
            )

        else:

            st.error(
                f"Risk Category: {risk_category}"
            )


        # ====================================================
        # MODEL SIGNAL
        # ====================================================

        st.write(
            f"**Model Signal:** {model_signal}"
        )


        st.caption(
            f"Model decision threshold: "
            f"{threshold:.2f}"
        )


        # ====================================================
        # 14. KEY RISK FACTORS
        # ====================================================

        st.divider()

        st.subheader(
            "Key Risk Factors"
        )

        risk_factors = []


        # Debt burden

        if debt_to_income > 1.0:

            risk_factors.append(
                "High debt burden: outstanding debt "
                "exceeds annual farm income."
            )

        elif debt_to_income > 0.60:

            risk_factors.append(
                "Elevated debt burden relative to "
                "annual farm income."
            )


        # Requested loan

        if loan_income_ratio > 0.80:

            risk_factors.append(
                "Requested loan is high relative "
                "to annual farm income."
            )

        elif loan_income_ratio > 0.60:

            risk_factors.append(
                "Requested loan represents a substantial "
                "share of annual farm income."
            )


        # Repayment

        if repayment_history_missing:

            risk_factors.append(
                "Previous repayment history is unavailable."
            )

        else:

            if repayment_rate < 0.60:

                risk_factors.append(
                    "Low historical repayment rate."
                )

            elif repayment_rate < 0.75:

                risk_factors.append(
                    "Moderate historical repayment rate."
                )

            if avg_delay_days > 30:

                risk_factors.append(
                    "High average repayment delay."
                )

            elif avg_delay_days > 15:

                risk_factors.append(
                    "Elevated average repayment delay."
                )


        # Weather

        if weather_risk_score > 0.60:

            risk_factors.append(
                "High weather-related agricultural risk."
            )

        elif weather_risk_score > 0.30:

            risk_factors.append(
                "Moderate weather-related agricultural risk."
            )


        # Irrigation

        if irrigation_binary == 0:

            risk_factors.append(
                "No irrigation availability reported."
            )


        # If no major risk factors

        if not risk_factors:

            risk_factors.append(
                "No major predefined risk factors "
                "were identified."
            )


        for factor in risk_factors:

            st.write(
                f"• {factor}"
            )


        # ====================================================
        # 15. ASSESSMENT SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "Assessment Summary"
        )


        summary_df = pd.DataFrame({

            "Metric": [
                "Adjusted Default-Risk Estimate",
                "Risk Category",
                "Model Signal",
                "Decision Threshold"
            ],

            "Result": [
                f"{probability * 100:.2f}%",
                risk_category,
                model_signal,
                f"{threshold:.2f}"
            ]
        })


        st.table(
            summary_df
        )


        # ====================================================
        # 16. DECISION SUPPORT NOTE
        # ====================================================

        st.info(
            """
            **Decision-support note**

            This assessment combines the trained machine-learning
            model with predefined financial and agricultural
            risk factors for this prototype.

            It is intended to support PACS credit assessment
            and should not be treated as an automatic lending
            decision.
            """
        )


    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.exception(e)


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.caption(
    "PACS Agricultural Credit-Risk Assessment • "
    "Machine Learning Decision-Support Prototype"
)