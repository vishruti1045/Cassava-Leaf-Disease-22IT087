import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

# Set page configuration
st.set_page_config(page_title="Petal", layout="centered", page_icon="🌿")

# Custom CSS for home page and design enhancements
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;  /* Light gray background for the main content */
        color: #333;  /* Dark text color for readability */
        padding: 20px;  /* Add padding to the main content */
    }
    .header {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;  /* Green color for the header */
        margin-bottom: 20px;
    }
    .intro-text {
        text-align: center;
        font-size: 18px;
        line-height: 1.8;
        color: #333;
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
        color: #4CAF50;
        margin-top: 40px;
    }
    .stSidebar {
        background-color: #f0f4f8;  /* Light grayish background for sidebar */
        padding: 20px;
        border-radius: 10px;
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
    st.markdown("""
        <div class="header">Welcome to Petal</div>
        <p class="intro-text">
            🌱 **Cassava Leaf Disease Detection** is a machine learning-based tool that helps farmers identify diseases in cassava leaves. 
            Upload an image of a cassava leaf, and our AI model will predict if the leaf shows any signs of disease or is healthy.
        </p>
    """, unsafe_allow_html=True)

    # Button to navigate to the image upload page
    if st.button('Upload Your Leaf Image'):
        st.session_state.page = 'Upload Image'  # Set the session state for the page

# Define a function for the image upload page
def upload_page():
    # Upload Cassava leaf image for disease prediction
    st.write("### Upload Cassava Leaf Image for Disease Prediction")
    uploaded_file = st.file_uploader("📷 Choose a file...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Cassava Leaf Image", use_column_width=True)
        st.write("📊 **Model Prediction**")

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

        # Display results with formatted text
        st.success(f"**Predicted Class:** {class_labels.get(predicted_class, 'Unknown')} ({predicted_class})")
        st.info(f"**Confidence:** {confidence:.2%}")

        # Provide additional explanation based on prediction
        if predicted_class != 4:
            st.warning("⚠️ **This leaf may show signs of disease.** Consider further analysis or treatment.")
        else:
            st.success("✅ **This leaf appears healthy!**")
    else:
        st.info("Please upload an image of a cassava leaf to proceed.")

    # Button to go back to the home page
    if st.button('Go Back to Home'):
        st.session_state.page = 'Home'  # Set the session state for the page

# Initialize session state if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar content (About the app and disease detection)
st.sidebar.title("📋 About Cassava and Disease Detection")
st.sidebar.write(
    """
    Cassava, the second-largest source of carbohydrates in Africa, is a vital crop for food security, 
    especially among smallholder farmers. This resilient starchy root thrives in harsh conditions but faces 
    significant challenges due to viral diseases that can severely impact crop yields.
    
    Leveraging machine learning, this app helps diagnose diseases in cassava leaves to support farmers with 
    early detection, enhancing food security across the region.
    """
)

# Render the selected page
if st.session_state.page == 'Home':
    home_page()
else:
    upload_page()

# Footer with additional resources
st.markdown("---")
st.markdown(
    "**Additional Resources:**\n"
    "- [Learn more about cassava diseases and treatment](https://plantvillage.psu.edu/topics/cassava-manioc/infos)\n"
    "- [App documentation](https://docs.streamlit.io/)"
)
