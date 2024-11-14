import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
import base64

# Set page configuration
st.set_page_config(page_title="Petal", layout="centered", page_icon="🌿")

# Custom CSS for home page and design enhancements
st.markdown("""
    <style>
    .main {
       background-color: #1c1e21; /* Dark background for main container */
    }
    .header {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #e6f4f1;  /* Light color for the header */
        margin-bottom: 20px;
    }
    .intro-text {
        text-align: center;
        font-size: 18px;    
        line-height: 1.8;
        color: #d9e2e1;  /* Light color for the intro text */
        margin-bottom: 30px;
    }
    .cta-button {
        text-align: center;
        margin-top: 20px;
    }
    .cta-button button {
        background-color: #4CAF50;  /* Green button */
        color: white;
        border-radius: 5px;
        padding: 15px 40px;
        font-size: 18px;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  /* Shadow for a 3D effect */
    }
    .cta-button button:hover {
        background-color: #45a049;  /* Darker green on hover */
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        margin-top: 40px;
    }
    .stSidebar {
        background-color: #2e2f31;  /* Dark gray background for sidebar */
        padding: 20px;
        border-radius: 10px;
        color: #ffffff;
    }
    .stSidebar h1, .stSidebar p {
        color: #d9e2e1; /* Light color for sidebar text */
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 15px 20px;
        width: 100%;
        font-size: 16px;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Function to encode image to base64 for CSS
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Custom CSS for background and styling
def set_glass_background(image_path):
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
        encoded_image = base64.b64encode(data).decode()
        background_css = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_image});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: #ffffff;
        }}
        .glass {{
            background: rgba(255, 255, 255, 0.2); /* Semi-transparent white background */
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 20px;
            margin-top: 50px;
        }}
        .header {{
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #e6f4f1;
            margin-bottom: 20px;
        }}
        .intro-text {{
            text-align: center;
            font-size: 18px;
            line-height: 1.8;
            color: white;
            margin-bottom: 30px;
        }}
        .cta-button {{
            text-align: center;
            margin-top: 20px;
        }}
        </style>
        """
        st.markdown(background_css, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading background image: {e}")

set_glass_background(r"C:\5-SGP\back-2.jpg")

# Load the pre-trained model
model = tf.keras.models.load_model("C:/model.keras", custom_objects={'BatchNormalization': BatchNormalization})

# Function to preprocess the image
def preprocess_image(image: Image.Image) -> np.array:
    img = image.resize((224, 224))  # Resize image to model's input size
    img_array = np.array(img) / 255.0  # Scale pixels to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define a function for the home page
def home_page():
    # Display the home page contents
    
    st.title("🌿 Petal - Cassava Leaf Disease Detection")
    st.markdown(
        "<div class='glass'>"
        "<div class='header'>Welcome to Petal</div>"
        "<p class='intro-text'>🌱 Petal-Cassava Leaf Disease Detection is a machine learning-based tool that helps farmers identify diseases in cassava leaves. "
        "Upload an image of a cassava leaf, and our AI model will predict if the leaf shows any signs of disease or is healthy.</p>"
        "</div>", unsafe_allow_html=True
    )

    # Button to navigate to the image upload page
    if st.button('Upload Your Leaf Image'):
        st.session_state.page = 'Upload Image'  # Set the session state for the page


    # Upload Cassava leaf image for disease prediction
def upload_page():
    st.markdown(
        "<div class='glass'>"
        "<h2>Upload Cassava Leaf Image for Disease Prediction</h2>"
        "</div>",
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("📷 Choose a file...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image in a glass container
        st.markdown(
            "<div class='glass'>"
            "<h4>Uploaded Cassava Leaf Image</h4>",
            unsafe_allow_html=True
        )
        st.image(uploaded_file, caption="Uploaded Cassava Leaf Image", use_container_width=True)


        # Display model prediction
        st.markdown("<h3>📊 Model Prediction</h3>", unsafe_allow_html=True)
        
        # Preprocess the image and make prediction
        image = Image.open(uploaded_file)
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Dictionary for class labels
        class_labels = {
            0: "Cassava Bacterial Blight (CBB)",
            1: "Cassava Brown Streak Disease (CBSD)",
            2: "Cassava Green Mottle (CGM)",
            3: "Cassava Mosaic Disease (CMD)",
            4: "Healthy Leaf"
        }

        # Display prediction result in a glass container
        st.markdown(
            f"<div class='glass'>"
            f"<h3 style='color:white;'>Predicted Class: {class_labels.get(predicted_class, 'Unknown')} ({predicted_class})</h3>"
            #f"<p style='color:#d9e2e1;'>Confidence: {confidence:.2%}</p>"
            "</div>", unsafe_allow_html=True
        )

        # Additional message based on the prediction
        if predicted_class != 4:
            st.markdown(
                f"<div class='glass' style='background-color: #ffdddd; color: #b30000; margin-top: 10px;'>"
                f"<strong>⚠️ This leaf may show signs of disease. Consider further analysis or treatment.</strong>"
                "</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='glass' style='background-color: #ddffdd; color: #006600; margin-top: 10px;'>"
                f"<strong>✅ This leaf appears healthy!</strong>"
                "</div>", unsafe_allow_html=True
            )

    else:
        st.markdown(
            "<div class='glass'>"
            "<strong>Please upload an image of a cassava leaf to proceed.</strong>"
            "</div>", unsafe_allow_html=True
        )

    # Button to go back to the home page
    if st.button('Go Back to Home'):
        st.session_state.page = 'Home'

# Initialize session state if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar content with glassmorphism effect
st.sidebar.markdown(
   
    "<h3>📋 About Cassava and Disease Detection</h3>"
    "<p>Cassava, the second-largest source of carbohydrates in Africa, is a vital crop for food security, "
    "especially among smallholder farmers. This resilient starchy root thrives in harsh conditions but faces "
    "significant challenges due to viral diseases that can severely impact crop yields.</p>"
    "<p>Leveraging machine learning, this app helps diagnose diseases in cassava leaves to support farmers with "
    "early detection, enhancing food security across the region.</p>"
    , unsafe_allow_html=True
)

# Render the selected page
if st.session_state.page == 'Home':
    home_page()
else:
    upload_page()

# Footer with additional resources
st.markdown("---")
st.markdown(
    "<div class='glass'>"
    "<h4>Additional Resources:</h4>"
    "<ul>"
    "<li><a href='https://plantvillage.psu.edu/topics/cassava-manioc/infos' target='_blank' style='color: white;'>Learn more about cassava diseases and treatment</a></li>"
    "<li><a href='https://docs.streamlit.io/' target='_blank' style='color: white;'>App documentation</a></li>"
    "</ul>"
    "</div>", 
    unsafe_allow_html=True
)
