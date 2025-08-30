import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os

# Load dataset
@st.cache_data
def load_data():
    if os.path.exists("Crop_recommendation.csv"):
        data = pd.read_csv("Crop_recommendation.csv")
        return data
    st.error("Dataset not found! Please place 'Crop_recommendation.csv' in the project directory.")
    return pd.DataFrame()

data = load_data()
if data.empty:
    st.stop()

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit UI
st.title("Smart Crop Recommendation System")
st.write("Enter the soil and environmental parameters below:")

N = st.number_input('Nitrogen (N)', min_value=0.0, max_value=140.0, value=50.0)
P = st.number_input('Phosphorus (P)', min_value=5.0, max_value=145.0, value=55.0)
K = st.number_input('Potassium (K)', min_value=5.0, max_value=205.0, value=45.0)
temperature = st.number_input('Temperature (°C)', min_value=8.0, max_value=43.0, value=25.0)
humidity = st.number_input('Humidity (%)', min_value=15.0, max_value=99.0, value=80.0)
ph = st.number_input('pH', min_value=3.5, max_value=9.9, value=6.5)
rainfall = st.number_input('Rainfall (mm)', min_value=20.0, max_value=300.0, value=110.0)
model_type = st.selectbox('Choose Model', ['Decision Tree', 'Naive Bayes'])

# Model Training
if model_type == 'Decision Tree':
    model = DecisionTreeClassifier(random_state=42)
else:
    model = GaussianNB()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Prediction
if st.button("Recommend Crop"):
    input_features = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(input_features)[0]
    st.success(f"Recommended Crop: *{prediction.capitalize()}*")
    st.write(f"Model accuracy on validation set: {acc*100:.2f}%")

st.markdown("---")
st.markdown("Use Case: Smart farming apps. [Agriculture India Dataset on Kaggle]")