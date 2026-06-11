import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Custom UI Styling

st.markdown("""
<style>

/* Entire Page */
.stApp {
    background: linear-gradient(
        135deg,
        #F4D0D9,
        #F4D0D9,
        #EFC7D2
    );
}

/* Main Text */
body, p, div, span, label {
    color: #000000 !important;
}

/* Main Heading */
h1 {
    color: #D63384 !important;
    text-align: center;
    font-weight: bold;
}

/* Sub Headings */
h2, h3 {
    color: #C2185B !important;
}

/* Radio Button Labels */
.stRadio label {
    color: #000000 !important;
}

/* Select Box Text */
.stSelectbox label {
    color: #000000 !important;
}

/* Dropdown Text */
div[data-baseweb="select"] {
    color: black !important;
    background-color: white;
    border-radius: 10px;
}

/* File Upload Box */
div[data-testid="stFileUploader"] {
    background-color: white;
    padding: 12px;
    border-radius: 15px;
}

/* Camera Box */
div[data-testid="stCameraInput"] {
    background-color: white;
    padding: 12px;
    border-radius: 15px;
}

/* Button */
.stButton > button {
    background-color: #FF69B4;
    color: white !important;
    border-radius: 12px;
    border: none;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

/* Button Hover */
.stButton > button:hover {
    background-color: #E75480;
    color: white !important;
}

/* Success Box Text */
[data-testid="stAlert"] {
    color: black !important;
}

/* Info Box */
.stInfo {
    color: black !important;
}

/* Warning Box */
.stWarning {
    color: black !important;
}

/* Sidebar (if added later) */
section[data-testid="stSidebar"] {
    background-color: #FFE4EC;
}
            /* Upload box text */
[data-testid="stFileUploader"] * {
    color: white !important;
}

/* Selectbox text */
[data-baseweb="select"] * {
    color: white !important;
}

/* Dropdown arrow */
[data-baseweb="select"] svg {
    fill: white !important;
}

/* Radio button text */
.stRadio * {
    color: black !important;
}

/* Uploaded filename */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: white !important;
}
/* Dropdown menu options */
ul[role="listbox"] li {
    color: white !important;
    background-color: #1F2230 !important;
}

/* Selected value */
[data-baseweb="select"] span {
    color: white !important;
}

/* Dropdown menu container */
div[role="listbox"] {
    background-color: #1F2230 !important;
}
/* Force dropdown text white */
[data-baseweb="popover"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Load trained model
model = tf.keras.models.load_model("acne_model.h5")

# Classes
classes = ["Mild", "Moderate", "Severe"]

# Recommendation Database
recommendations = {

    ("Mild", "Oily"): {
        "use": [
            "Niacinamide",
            "Salicylic Acid",
            "Zinc PCA",
            "Oil-Free Moisturizer"
        ],
        "avoid": [
            "Lanolin",
            "Mineral Oil",
            "Cocoa Butter",
            "Heavy Oils"
        ],
        "routine": """
🌞 Morning
• Gentle Foaming Cleanser
• Niacinamide Serum
• Oil-Free Moisturizer
• Sunscreen SPF 30+

🌙 Night
• Salicylic Acid Cleanser
• Niacinamide Serum
• Lightweight Moisturizer
"""
    },

    ("Moderate", "Oily"): {
        "use": [
            "Salicylic Acid",
            "Niacinamide",
            "Azelaic Acid",
            "Zinc PCA"
        ],
        "avoid": [
            "Lanolin",
            "Mineral Oil",
            "Heavy Oils",
            "Comedogenic Makeup"
        ],
        "routine": """
🌞 Morning
• Salicylic Acid Cleanser
• Niacinamide
• Oil-Free Moisturizer
• Sunscreen SPF 30+

🌙 Night
• Gentle Cleanser
• Azelaic Acid
• Lightweight Moisturizer
"""
    },

    ("Severe", "Oily"): {
        "use": [
            "Benzoyl Peroxide",
            "Adapalene",
            "Niacinamide",
            "Dermatologist Consultation"
        ],
        "avoid": [
            "Heavy Oils",
            "Harsh Scrubs",
            "Lanolin",
            "Mineral Oil"
        ],
        "routine": """
🌞 Morning
• Gentle Cleanser
• Benzoyl Peroxide
• Moisturizer
• Sunscreen SPF 30+

🌙 Night
• Gentle Cleanser
• Adapalene
• Moisturizer
"""
    },

    ("Mild", "Dry"): {
        "use": [
            "Ceramides",
            "Hyaluronic Acid",
            "Niacinamide",
            "Glycerin"
        ],
        "avoid": [
            "Alcohol-Based Toners",
            "Harsh Exfoliants",
            "Strong Acids"
        ],
        "routine": """
🌞 Morning
• Gentle Hydrating Cleanser
• Hyaluronic Acid
• Ceramide Moisturizer
• Sunscreen SPF 30+

🌙 Night
• Gentle Cleanser
• Niacinamide
• Ceramide Moisturizer
"""
    },

    ("Moderate", "Dry"): {
        "use": [
            "Ceramides",
            "Azelaic Acid",
            "Niacinamide",
            "Hyaluronic Acid"
        ],
        "avoid": [
            "Alcohol-Based Products",
            "Strong Salicylic Acid",
            "Harsh Exfoliants"
        ],
        "routine": """
🌞 Morning
• Gentle Cleanser
• Hyaluronic Acid
• Ceramide Moisturizer
• Sunscreen SPF 30+

🌙 Night
• Azelaic Acid
• Ceramide Moisturizer
"""
    },

    ("Severe", "Dry"): {
        "use": [
            "Ceramides",
            "Barrier Repair Cream",
            "Gentle Cleanser",
            "Dermatologist Consultation"
        ],
        "avoid": [
            "Strong Retinoids",
            "Over-Exfoliation",
            "Alcohol Products"
        ],
        "routine": """
🌞 Morning
• Gentle Cleanser
• Barrier Repair Cream
• Sunscreen SPF 30+

🌙 Night
• Gentle Cleanser
• Ceramide Moisturizer
"""
    },

    ("Mild", "Sensitive"): {
        "use": [
            "Centella Asiatica",
            "Ceramides",
            "Niacinamide",
            "Panthenol"
        ],
        "avoid": [
            "Fragrance",
            "Essential Oils",
            "Strong Acids"
        ],
        "routine": """
🌞 Morning
• Gentle Cleanser
• Centella Serum
• Moisturizer
• Sunscreen SPF 30+

🌙 Night
• Gentle Cleanser
• Niacinamide
• Ceramide Moisturizer
"""
    },

    ("Moderate", "Sensitive"): {
        "use": [
            "Azelaic Acid",
            "Niacinamide",
            "Ceramides",
            "Panthenol"
        ],
        "avoid": [
            "Benzoyl Peroxide",
            "Fragrance",
            "Harsh Scrubs"
        ],
        "routine": """
🌞 Morning
• Gentle Cleanser
• Ceramide Moisturizer
• Sunscreen SPF 30+

🌙 Night
• Azelaic Acid
• Ceramide Moisturizer
"""
    },

    ("Severe", "Sensitive"): {
        "use": [
            "Barrier Repair Cream",
            "Ceramides",
            "Gentle Cleanser",
            "Dermatologist Consultation"
        ],
        "avoid": [
            "Strong Acids",
            "Fragrance",
            "Over-Exfoliation"
        ],
        "routine": """
🌞 Morning
• Gentle Cleanser
• Barrier Repair Cream
• Sunscreen SPF 30+

🌙 Night
• Gentle Cleanser
• Ceramide Moisturizer
"""
    },

    ("Mild", "Combination"): {
        "use": [
            "Niacinamide",
            "Salicylic Acid",
            "Hyaluronic Acid",
            "Lightweight Moisturizer"
        ],
        "avoid": [
            "Heavy Oils",
            "Lanolin"
        ],
        "routine": """
🌞 Morning
• Gentle Cleanser
• Niacinamide
• Lightweight Moisturizer
• Sunscreen SPF 30+

🌙 Night
• Salicylic Acid
• Moisturizer
"""
    },

    ("Moderate", "Combination"): {
        "use": [
            "Salicylic Acid",
            "Niacinamide",
            "Azelaic Acid",
            "Zinc PCA"
        ],
        "avoid": [
            "Heavy Creams",
            "Lanolin",
            "Harsh Scrubs"
        ],
        "routine": """
🌞 Morning
• Salicylic Acid Cleanser
• Niacinamide
• Moisturizer
• Sunscreen SPF 30+

🌙 Night
• Azelaic Acid
• Moisturizer
"""
    },

    ("Severe", "Combination"): {
        "use": [
            "Benzoyl Peroxide",
            "Adapalene",
            "Gentle Cleanser",
            "Dermatologist Consultation"
        ],
        "avoid": [
            "Heavy Oils",
            "Multiple Active Ingredients",
            "Harsh Scrubs"
        ],
        "routine": """
🌞 Morning
• Gentle Cleanser
• Benzoyl Peroxide
• Moisturizer
• Sunscreen SPF 30+

🌙 Night
• Adapalene
• Moisturizer
"""
    }
}

# UI
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap" rel="stylesheet">

<h1 style="
font-family:'Dancing Script', cursive;
color:#B03060;
text-align:center;
font-size:85px;
margin-bottom:-15px;
">
 Skinnova
</h1>

<h3 style="
text-align:center;
color:#8E4585;
font-family:'Segoe UI';
margin-top:0px;
">
AI-Powered Acne Detection & Personalized Skincare Guidance
</h3>
""", unsafe_allow_html=True)

st.markdown("---")

# IMAGE SOURCE

st.subheader("📸 Upload or Capture Facial Image")

image_source = st.radio(
    "Select an option",
    ["Upload Image", "Use Camera"]
)

uploaded_file = None

if image_source == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload Facial Image",
        type=["jpg", "jpeg", "png"]
    )

elif image_source == "Use Camera":
    uploaded_file = st.camera_input(
        "Take a Picture"
    )

# SKIN TYPE

skin_type = st.selectbox(
    "Select Skin Type",
    ["Oily", "Dry", "Sensitive", "Combination"]
)

# PREDICTION

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, width="stretch")

    if st.button("Analyze Skin"):

        img = image.resize((224, 224))

        img_array = np.array(img) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 Analyzing your skin..."):
         prediction = model.predict(img_array)

        result_index = np.argmax(prediction)

        result = classes[result_index]

        confidence = prediction[0][result_index] * 100

        st.subheader("Acne Severity")

        if result == "Mild":
         st.success("🟢 Mild Acne")
         st.write("Small number of pimples with limited inflammation.")

        elif result == "Moderate":
         st.warning("🟡 Moderate Acne")
         st.write("Noticeable inflammatory acne requiring active treatment.")

        else:
         st.error("🔴 Severe Acne")
         st.write("Extensive acne lesions. Dermatologist consultation recommended.")

        st.info(f"Confidence Score: {confidence:.2f}%")

        info = recommendations.get((result, skin_type))

        if info:

         col1, col2 = st.columns(2)

         with col1:

            st.subheader("✨ Recommended Ingredients")

            for item in info["use"]:
              st.write(f"✅ {item}")

         with col2:

            st.subheader("🚫 Ingredients To Avoid")

            for item in info["avoid"]:
              st.write(f"❌ {item}")

        st.subheader("🧴 Suggested Skincare Routine")

        st.write(info["routine"])

        st.warning(
            "This tool provides skincare guidance and is not a medical diagnosis. Consult a dermatologist for severe or persistent acne."
        )

        st.markdown("---")

st.markdown(
"""
<div style='text-align:center;'>

 Healthy skin is a reflection of overall wellness.

<br>

 Skinnova | AI-Powered Skincare Guidance

</div>
""",
unsafe_allow_html=True
)