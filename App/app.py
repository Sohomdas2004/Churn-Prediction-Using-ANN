
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow
from tensorflow import keras



st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Model" / "classifier.keras"
PREPROCESSOR_PATH = BASE_DIR / "Model" / "preprocessor.pkl"


st.title("🏦 Bank Customer Churn Prediction")

st.write(
    "Enter the customer's information below and the trained "
    "ANN model will estimate the probability of churn."
)

st.info(
    "This application is a deep-learning demonstration. "
    "The prediction is based on the trained model and should "
    "not be treated as a guaranteed outcome."
)



with st.form("prediction_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)


    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
            step=1,
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"],
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40,
            step=1,
        )

        tenure = st.number_input(
            "Tenure (Years)",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
        )


    with col2:

        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            max_value=300000.0,
            value=75000.0,
            step=1000.0,
        )

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=1,
            step=1,
        )

        has_cr_card = st.selectbox(
            "Has Credit Card?",
            [0, 1],
            format_func=lambda x: (
                "No (0)" if x == 0 else "Yes (1)"
            ),
        )

        is_active_member = st.selectbox(
            "Is Active Member?",
            [0, 1],
            format_func=lambda x: (
                "No (0)" if x == 0 else "Yes (1)"
            ),
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            max_value=250000.0,
            value=100000.0,
            step=1000.0,
        )

    submitted = st.form_submit_button(
        "🔍 Predict",
        use_container_width=True,
        type="primary",
    )


if submitted:

    try:


        if not MODEL_PATH.exists():

            st.error(
                f"ANN model was not found:\n\n{MODEL_PATH}"
            )

            st.stop()

        

        if not PREPROCESSOR_PATH.exists():

            st.error(
                f"Preprocessor was not found:\n\n"
                f"{PREPROCESSOR_PATH}"
            )

            st.stop()

        

        model = tensorflow.keras.models.load_model(MODEL_PATH)

      

        preprocessor = joblib.load(PREPROCESSOR_PATH)


        input_data = pd.DataFrame(
            {
                "CreditScore": [credit_score],
                "Geography": [geography],
                "Gender": [gender],
                "Age": [age],
                "Tenure": [tenure],
                "Balance": [balance],
                "NumOfProducts": [num_products],
                "HasCrCard": [has_cr_card],
                "IsActiveMember": [is_active_member],
                "EstimatedSalary": [estimated_salary],
            }
        )

       
        model_input = preprocessor.transform(input_data)

     
        model_input = np.asarray(model_input)

     

        prediction_output = model.predict(
            model_input,
            verbose=0,
        )

        prediction_array = np.asarray(
            prediction_output
        )

       
        probability = float(
            prediction_array.reshape(-1)[0]
        )

        
        probability = max(
            0.0,
            min(1.0, probability)
        )

 

        st.divider()

        st.subheader("Prediction Result")

        st.metric(
            "Estimated Probability of Churn",
            f"{probability * 100:.2f}%",
        )

        # -------------------------------------------------
        # CHURN CLASS
        # -------------------------------------------------

        if probability >= 0.5:

            st.error(
                "Model Prediction: Higher Predicted Churn Risk"
            )

        else:

            st.success(
                "Model Prediction: Lower Predicted Churn Risk"
            )

        st.caption(
            "A probability of 0.50 or higher is classified "
            "as churn using the default binary classification "
            "threshold."
        )

       

        with st.expander("View Customer Input"):

            st.dataframe(
                input_data,
                use_container_width=True,
            )


        with st.expander("View Processed Model Input"):

            processed_df = pd.DataFrame(
                model_input
            )

            st.dataframe(
                processed_df,
                use_container_width=True,
            )

    

    except FileNotFoundError as e:

        st.error(
            f"Required file was not found:\n\n{e}"
        )

    except ValueError as e:

        st.error(
            f"Preprocessing error:\n\n{e}"
        )

        st.info(
            "Make sure the columns used here match the "
            "columns used when preprocessor.pkl was trained."
        )

    except Exception as e:

        st.error(
            f"An error occurred while making the prediction:\n\n{e}"
        )

        st.exception(e)




