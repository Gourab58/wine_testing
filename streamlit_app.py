import streamlit as st
import numpy as np
import warnings
import joblib
warnings.filterwarnings('ignore')

# Load model and scaler
model = joblib.load(open("wine_quality_model.pkl", "rb"))
scaler = joblib.load(open("wine_quality_scaler.pkl", "rb"))

# Mapping function for prediction
def pred(value):
    if value == 0:
        return 'Excellent'
    elif value == 1:
        return 'Good'
    elif value == 2:
        return 'Average'
    else:
        return 'Poor'

# Streamlit App Interface
def main():
    st.title("Wine Quality Prediction")

    # Collect user input
    flavanoids = st.number_input('Flavanoids:', min_value=0.0, max_value=10.0, step=0.1)
    color_intensity = st.number_input('Color Intensity:', min_value=0.0, max_value=10.0, step=0.1)
    proline = st.number_input('Proline:', min_value=0.0, max_value=2000.0, step=1.0)
    ash = st.number_input('Ash:', min_value=0.0, max_value=5.0, step=0.1)
    alcohol = st.number_input('Alcohol:', min_value=0.0, max_value=15.0, step=0.1)

    # When the user clicks the predict button
    if st.button('Predict'):
        try:
            # Prepare input for model
            raw_data = np.array([[flavanoids, color_intensity, proline, ash, alcohol]])
            scaled_data = scaler.transform(raw_data)
            prediction = model.predict(scaled_data)

            # Map prediction to label
            final_prediction = pred(prediction[0])
            st.success(f"Wine Quality is: {final_prediction}")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()