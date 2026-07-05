import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Solar Panel Defect Classifier",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Constants & Descriptions
# ==========================================
CLASSES = [
    "Bird-drop", "Clean", "Dusty", 
    "Electrical-damage", "Physical-damage", "Snow-Covered"
]

DESCRIPTIONS = {
    "Bird-drop": "Bird droppings reduce sunlight reaching the cells and should be cleaned.",
    "Clean": "The panel surface is clear and operating under normal conditions.",
    "Dusty": "Dust accumulation can significantly reduce energy generation.",
    "Electrical-damage": "Electrical faults may indicate hotspots or circuit issues requiring inspection.",
    "Physical-damage": "Cracks, broken glass, or structural damage may affect performance and safety.",
    "Snow-Covered": "Snow blocks sunlight and should be removed when safe to restore efficiency."
}

# ==========================================
# Model Loading
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_effnet_finetune.h5")

# ==========================================
# Sidebar UI
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3105/3105807.png", width=120)
    st.title("Solar Panel AI")
    
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About
    This application uses a fine-tuned **EfficientNet CNN** model to identify different types of defects present on solar panels.
    """)
    
    st.markdown("### 🔍 Detectable Classes")
    for cls in CLASSES:
        st.write(f"✔️ {cls}")
        
    st.markdown("---")
    st.caption("Developed using TensorFlow, EfficientNet, and Streamlit.")

# ==========================================
# Main Content Area
# ==========================================
st.title("☀️ Solar Panel Defect Detection")
st.markdown("### AI-Powered Inspection using EfficientNet CNN")
st.write("Upload an image of a solar panel to automatically detect defects and receive confidence scores.")

# Load the model
try:
    with st.spinner("Loading model infrastructure..."):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload a solar panel image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("---")
    
    # Create two main columns for layout
    col_img, col_results = st.columns([1, 1.2], gap="large")
    
    with col_img:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
        
    # Preprocess and Predict
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))
    
    with st.spinner("Analyzing the panel..."):
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = CLASSES[predicted_idx]

    with col_results:
        st.subheader("Analysis Results")
        
        # Display Status Alert
        if predicted_class == "Clean":
            st.success(f"**Status: Healthy** — The panel appears to be in good condition!")
        else:
            st.warning(f"**Status: Attention Required** — A defect or contamination was detected.")
            
        # Metrics Row
        m1, m2 = st.columns(2)
        m1.metric("Top Prediction", predicted_class)
        m2.metric("Confidence", f"{confidence:.2%}")
        
        # Display Description
        st.info(f"**Diagnosis:** {DESCRIPTIONS[predicted_class]}")
        st.progress(float(confidence))

    # ==========================================
    # Detailed Data Tabs
    # ==========================================
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Prediction Distribution", "🏆 Top Predictions", "🧠 About the Model"])
    
    with tab1:
        st.write("Visual breakdown of the model's confidence across all categories:")
        df = pd.DataFrame({"Class": CLASSES, "Probability": predictions[0]})
        st.bar_chart(df.set_index("Class"))
        
    with tab2:
        st.write("Highest probability matches:")
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        
        for i, idx in enumerate(top_indices):
            class_name = CLASSES[idx]
            prob = predictions[0][idx]
            if i == 0:
                st.markdown(f"🥇 **{class_name}** - `{prob:.2%}`")
            elif i == 1:
                st.markdown(f"🥈 **{class_name}** - `{prob:.2%}`")
            else:
                st.markdown(f"🥉 **{class_name}** - `{prob:.2%}`")
                
        with st.expander("View raw probabilities for all classes"):
            for i, prob in enumerate(predictions[0]):
                st.code(f"{CLASSES[i]:<20} {prob:.2%}")

    with tab3:
        st.write("""
        This application uses **EfficientNet**, a state-of-the-art Convolutional Neural Network (CNN) 
        architecture developed by Google. EfficientNet achieves high accuracy while using fewer parameters 
        by uniformly scaling Network Depth, Network Width, and Input Resolution.
        """)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Convolutional Neural Networks (CNN)**
            * Automatically learn edges, shapes, and textures.
            * Composed of Convolution, Pooling, and Fully Connected layers.
            * Standard architecture for modern image classification.
            """)
        with col_b:
            st.markdown("""
            **Why EfficientNet?**
            * Higher accuracy with lower computational cost.
            * Faster inference times.
            * Compound scaling method balances width, depth, and resolution.
            * Excellent base for transfer learning on custom datasets.
            """)