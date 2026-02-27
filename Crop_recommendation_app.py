import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os

import streamlit as st
import pandas as pd
import streamlit as st
# ... other imports
# Crop image mapping (use online images or local ones)
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
# 1. SET WIDE MODE (Removes the narrow center column)
st.set_page_config(page_title="Smart Crop AI", layout="wide", initial_sidebar_state="collapsed")

def apply_dynamic_full_width_theme():
    st.markdown(
        """
        <style>
        /* 2. REMOVE CONTAINER LIMITS */
        .block-container {
            max-width: 100% !important;
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
            padding-left: 5rem !important;
            padding-right: 5rem !important;
        }

        /* 3. DYNAMIC BACKGROUND */
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                        url("https://images.unsplash.com/photo-1560493676-04071c5f467b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
            background-size: cover;
            background-attachment: fixed;
        }

        /* 4. ANIMATED TITLE */
        .dynamic-title {
            transition: transform 0.3s ease-in-out;
            cursor: default;
        }
        .dynamic-title:hover {
            transform: scale(1.02);
            color: #8bc34a !important;
        }

        /* 5. PULSE ANIMATION FOR RESULT */
        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(139, 195, 74, 0.7); }
            70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(139, 195, 74, 0); }
            100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(139, 195, 74, 0); }
        }
        .result-card {
            animation: pulse 2s infinite;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 2px solid #8bc34a;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }

        /* Crop Card Animation */
@keyframes slideFade {
    from {
        opacity: 0;
        transform: translateY(40px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.crop-card {
    animation: slideFade 1s ease forwards;
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    border: 2px solid #8bc34a;
    margin-top: 30px;
}

.crop-card img {
    width: 280px;
    height: 180px;
    border-radius: 15px;
    object-fit: cover;
    margin-bottom: 15px;
}
        </style>
        """,
        unsafe_allow_html=True
    )

apply_dynamic_full_width_theme()

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
    prediction = model.predict(input_features)[0].lower()

    crop_image = CROP_IMAGES.get(prediction, 
        "https://images.unsplash.com/photo-1501004318641-b39e6451bec6")

    st.markdown(f"""
        <div class="crop-card">
            <h3 style="color:white;">🌱 Recommended Crop</h3>
            <img src="{crop_image}" alt="{prediction}">
            <h1 style="color:#8bc34a; font-size:45px;">{prediction.upper()}</h1>
            <p style="color:white; opacity:0.8;">
                Model Confidence: {acc*100:.2f}%
            </p>
        </div>
    """, unsafe_allow_html=True)
