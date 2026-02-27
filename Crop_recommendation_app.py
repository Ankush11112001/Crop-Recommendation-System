import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os
def apply_tech_farming_theme():
    st.markdown(
        """
        <style>
        /* 1. The Background */
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                        url("https://images.unsplash.com/photo-1560493676-04071c5f467b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80") !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
        }

        /* 2. The Main Container (Glass effect) */
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(12px) !important;
            padding: 50px !important;
            border-radius: 25px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            margin-top: 50px !important;
        }

        /* 3. Text Visibility */
        h1, h2, h3, p, label, .stMarkdown {
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
        }

        /* 4. Inputs (Making them solid so they don't fade) */
        .stNumberInput input, .stSelectbox div {
            background-color: white !important;
            color: black !important;
            border-radius: 8px !important;
        }
        
        /* 5. The Button */
        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            border: none !important;
            padding: 10px 24px !important;
            font-size: 18px !important;
            font-weight: bold !important;
            border-radius: 12px !important;
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

apply_tech_farming_theme()
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
