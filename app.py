import streamlit as st
import numpy as np
import pickle

# Loading the trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
std_scaler = pickle.load(open('standscaler.pkl', 'rb'))


# Crop dictionary used for decoding the outcome of prediction.
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 20: 'kidneybeans',
    21: 'chickpea', 22: 'coffee'
}

# Title of the Streamlit app
st.title("Crop Recommendation")

# Input Columns
N = st.number_input('Nitrogen content (N)', min_value=0, max_value=100, value=50)
P = st.number_input('Phosphorus content (P)', min_value=0, max_value=100, value=50)
K = st.number_input('Potassium content (K)', min_value=0, max_value=100, value=50)
temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input('pH value', min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, value=100.0)

# Prediction function
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled_features = std_scaler.transform(minmax_scaler.transform(features))
    prediction = model.predict(scaled_features).reshape(1, -1)[0][0]
    return crop_dict[prediction]

# Button for prediction
if st.button('Recommend Crop'):
    crop = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
    st.success(f'The recommended crop is: {crop}')
