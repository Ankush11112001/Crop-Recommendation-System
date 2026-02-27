import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os

# 1. Page Config
st.set_page_config(page_title="Smart Crop AI", layout="wide", initial_sidebar_state="collapsed")

# 2. Crop image mapping
CROP_IMAGES = {
    "rice": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR4FArXQuGECLU50-0bBjXg53jJKir-pAR2wQ&s",
    "maize": "https://cdn.britannica.com/36/167236-050-BF90337E/Ears-corn.jpg",
    "chickpea": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYRax3XES8P0rpQ6JPqiAiDVnwFecEhg1QUg&s",
    "kidneybeans": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT-4cI_0iwz7t11MPwse54LUWH-XhDm2O8-Kw&s",
    "pigeonpeas": "https://sustainableholly.com/wp-content/uploads/2025/06/pigeon-pea-green-pods-edited.jpg",
    "mothbeans": "https://harikrushnaexports.com/wp-content/uploads/2024/03/mothbean-500x500-1.webp",
    "mungbean": "https://www.epicgardening.com/wp-content/uploads/2021/10/A-close-up-shot-of-several-pods-of-legumes-alongside-their-leaves-showcasing-the-mung-bean.jpg",
    "blackgram": "https://cpimg.tistatic.com/10577159/b/4/Black-Gram..jpg",
    "lentil": "https://organicseeds.top/_bl/2/14017989.jpg",
    "pomegranate": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTvos2JPAkMavZxm5bEEeiWYrRjRqv7TQMTyw&s",
    "banana": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQHCrKNAZB4q7jEh6NyEvUF8OjWRjWm5E7izw&s",
    "mango": "https://media.istockphoto.com/id/1091407688/photo/ripening-mangoes-hanging-from-the-tree.jpg?s=612x612&w=0&k=20&c=yAypI7xxT4A_aC2PV_HQQbj844hjVOQclfphSFEJnLg=",
    "grapes": "https://www.foodrepublic.com/img/gallery/15-types-of-grapes-to-know-eat-and-drink/l-intro-1743188162.jpg",
    "watermelon": "https://cdn.pixabay.com/photo/2017/08/28/21/54/melon-2691415_1280.jpg",
    "muskmelon": "https://bombayseeds.com/cdn/shop/files/Musk_melon_seeds.jpg?v=1729232739",
    "apple": "https://brownchinarkashmir.com/wp-content/uploads/2024/12/Kashmir_Apple_Season_brown_chinar_kashmir-1.jpg.webp",
    "orange": "https://draxe.com/wp-content/uploads/2018/03/DrAxeOrangeNutrition_Thumbnail.jpg",
    "papaya": "https://www.dreamfoodscaribe.com/wp-content/uploads/2024/07/papaya-fruit.webp",
    "coconut": "https://media.istockphoto.com/id/1407981572/photo/coconut-tree-at-coconut-farm.jpg?s=612x612&w=0&k=20&c=Mheo-LyMZpWcIVGl2Awh-8aK-MNgGTJuH78v4ChvfG0=",
    "cotton": "https://t4.ftcdn.net/jpg/06/84/31/79/360_F_684317966_Pn9qU1DEfW5zpwoj25znJ1i0VdaOM2Px.jpg",
    "jute": "https://thumbs.dreamstime.com/b/jute-plants-field-jute-cultivation-assam-india-jute-plants-field-jute-cultivation-assam-india-jute-fiber-258026749.jpg",
    "coffee": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTe4yrLCeFKzilrAmAk5JIigzAByi9j6acYEA&s"
}

def apply_final_mobile_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                        url("https://images.unsplash.com/photo-1560493676-04071c5f467b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80") !important;
            background-size: cover !important;
            background-attachment: fixed !important;
        }

        .main .block-container {
            background-color: transparent !important;
            max-width: 900px !important;
            margin: auto !important;
            padding: 2rem 1rem !important;
        }

        h1, h2, h3, p, label, .stMarkdown p {
            color: #ffffff !important;
            background-color: transparent !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
            text-align: center;
        }

        .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: white !important;
            color: black !important;
            border-radius: 8px !important;
        }

        .crop-card {
            background: rgba(255,255,255,0.12);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            border: 2px solid #8bc34a;
            margin: 30px auto;
            max-width: 450px;
        }

        .crop-card img {
            width: 100% !important;
            max-width: 300px !important;
            height: auto !important;
            border-radius: 15px;
            margin: 15px auto;
            display: block;
        }

        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 12px !important;
            width: 100% !important;
            height: 3em;
            font-size: 1.2rem !important;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

apply_final_mobile_theme()

# Load dataset
@st.cache_data
def load_data():
    if os.path.exists("Crop_recommendation.csv"):
        return pd.read_csv("Crop_recommendation.csv")
    return pd.DataFrame()

data = load_data()
if data.empty:
    st.error("Dataset not found! Please check the filename.")
    st.stop()

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit UI
st.title("Smart Crop Recommendation System")
st.write("Fill in the soil details below:")

# Better mobile input layout
col1, col2 = st.columns(2)
with col1:
    N = st.number_input('Nitrogen (N)', 0.0, 140.0, 50.0)
    P = st.number_input('Phosphorus (P)', 5.0, 145.0, 55.0)
    K = st.number_input('Potassium (K)', 5.0, 205.0, 45.0)
with col2:
    temp = st.number_input('Temperature (°C)', 8.0, 43.0, 25.0)
    hum = st.number_input('Humidity (%)', 15.0, 99.0, 80.0)
    ph = st.number_input('pH', 3.5, 9.9, 6.5)

rainfall = st.number_input('Rainfall (mm)', 20.0, 300.0, 110.0)
model_type = st.selectbox('Choose Model', ['Decision Tree', 'Naive Bayes'])

# Model Training
model = DecisionTreeClassifier(random_state=42) if model_type == 'Decision Tree' else GaussianNB()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Prediction Button (Placed outside columns for visibility)
if st.button("Recommend My Crop"):
    prediction = model.predict([[N, P, K, temp, hum, ph, rainfall]])[0].lower()
    crop_image = CROP_IMAGES.get(prediction, "https://images.unsplash.com/photo-1501004318641-b39e6451bec6")

    st.markdown(f"""
        <div class="crop-card">
            <h3 style="margin-bottom:0;">🌱 Recommended Crop</h3>
            <img src="{crop_image}">
            <h1 style="color:#8bc34a; margin-top:0;">{prediction.upper()}</h1>
            <p style="font-size:0.9rem; opacity:0.8;">Model Confidence: {acc*100:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)
