# app.py

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

# Import all our functions
from src.hcr_module import (
    get_tesseract_transcription, 
    load_trocr_model, 
    get_trocr_transcription, 
    post_process_text
)

from src.preprocessing_module import preprocess_for_ocr, preprocess_for_trocr, preprocess_for_graphology
from src.graphology_module import (
    extract_graphological_features, 
    get_personality_profile,
    create_spider_chart
)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InkSight: Handwriting Analysis",
    page_icon="✒️",
    layout="wide",
)

# --- CUSTOM CSS STYLING ---
st.markdown(""" 
<style>
/* 4. Full Page Background */
body {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    background-attachment: fixed;
}

/* 4. Main content "card" */
.main .block-container {
    background: rgba(255, 255, 255, 0.9); /* More opaque */
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

/* 2. Styled Title Card */
.title-card {
    background: linear-gradient(135deg, #007bff 0%, #333399 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    margin-bottom: 2rem;
}
.title-card h1 {
    color: white;
    font-size: 2.5em;
    margin: 0;
}

/* 1. Flexible Text Area */
.text-display-box {
    background-color: #f9f9f9;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    min-height: 150px;
    max-height: 400px;
    overflow-y: auto;
    font-family: 'Source Sans Pro', sans-serif;
    line-height: 1.6;
    color: #000;
}

/* Styled Tab Subheader */
.tab-subheader {
    background-color: #f0f2f6;
    border-left: 5px solid #007bff;
    padding: 0.75rem 1rem;
    border-radius: 5px;
    margin-bottom: 1rem;
}
.tab-subheader h3 {
    margin: 0;
    color: #003366;
}

/* Component Styling */
[data-testid="stInfo"] {
    background-color: #e8f1ff;
    border: 1px solid #007bff;
    border-radius: 8px;
}
[data-testid="stFileUploader"] {
    background-color: #fdfdfd;
    padding: 1rem;
    border-radius: 10px;
    border: 2px dashed #007bff;
}
[data-testid="stRadio"] {
    background-color: #fdfdfd;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid rgba(0, 123, 255, 0.3);
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* --- CORRECTED: Center-align tabs and remove bottom border --- */

/* This is the new, safer selector for the tab bar */
[data-testid="stTabBar"] {
    display: flex;
    justify-content: center; /* This centers the buttons */
}

/* This styles the tab buttons */
[data-testid="stTabs"] button[role="tab"] {
    font-size: 1.1em;
    font-weight: bold;
    color: #333;
    background: #f0f2f6;
    border: 1px solid #ccc;
    border-radius: 8px 8px 8px 8px; /* Rounded all corners */
    margin: 0 5px; /* Add space between buttons */
    transition: all 0.3s ease;
    padding: 0.5rem 1rem;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    background: #e0e6ed;
    border-color: #007bff;
}
/* This styles the *selected* tab button */
[data-testid="stTabs"] button[aria-selected="true"] {
    color: white;
    background: #007bff; /* Make the selected button solid blue */
    border: 1px solid #007bff;
    box-shadow: 0 2px 5px rgba(0,0,0,0.15);
}

/* This styles the content panel below the tabs */
[data-testid="stTabs"] div[role="tabpanel"] {
    background: #ffffff;
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid #ddd;
    margin-top: 10px; /* Add space above the content panel */
}
</style>
""", unsafe_allow_html=True)


# --- APP TITLE ---
st.markdown(
    '<div class="title-card"><h1>✒️ InkSight: AI-Powered Handwriting Analysis</h1></div>', 
    unsafe_allow_html=True
)
#--- FILE UPLOADER & MAIN LOGIC ---
st.info("Welcome to InkSight ! Please upload a handwriting image to begin transcription and analysis.")
uploaded_file = st.file_uploader("**Choose a handwriting image...**", type=["jpg", "jpeg", "png"])

#--- MODE SELECTOR ---
#Add a radio button to choose the OCR engine
mode = st.radio(
    "**Select Handwriting Type (for Transcription):**",
    ["**(Print / Clear Handwriting)** (Fast, good for simple text - Tesseract)", 
     "**(Cursive / Complex Handwriting)** (Slower, best for accuracy - TrOCR)"],
    index=0) # Default to 'Print'
st.markdown("---")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Handwriting', width='content')
    
    with st.spinner('Analyzing your handwriting... This might take a moment.'):
        # --- ORGANIZE RESULTS INTO TABS ---
        tab1, tab2, tab3 = st.tabs([
            "✍️ Digital Transcription", 
            "🧠 Personality Insights", 
            "⚙️ Extracted Features"
        ])

        # --- Tab 1: Digital Transcription ---
        with tab1:
            if "Cursive" in mode:
                # Use TrOCR
                st.markdown(
                    '<div class="tab-subheader"><h3>Cursive Model Transcription</h3></div>', 
                    unsafe_allow_html=True
                )
                try:
                    processor, model = load_trocr_model()
                    transcribed_text = get_trocr_transcription(processor, model, img_array)
                    corrected_text = post_process_text(transcribed_text)
                    st.markdown(
                    '<div class="tab-subheader"><h3>Result: </h3></div>', 
                    unsafe_allow_html=True
                )
                    st.markdown(
                        f'<div class="text-display-box">{corrected_text}</div>', 
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error loading or running TrOCR model: {e}")
                    st.error("This model requires a one-time download, please ensure you are connected to the internet.")

            else:
                # Use Tesseract (Print Mode)
                st.markdown(
                    '<div class="tab-subheader"><h3>Print Model Transcription</h3></div>', 
                    unsafe_allow_html=True
                )
                st.write("**Applying preprocessing for Tesseract...**")
                preprocessed_image = preprocess_for_ocr(img_array)
                st.image(preprocessed_image, caption='Preprocessed for OCR', use_container_width=False)

                transcribed_text = get_tesseract_transcription(preprocessed_image)
                corrected_text = post_process_text(transcribed_text)
                st.markdown(
                    '<div class="tab-subheader"><h3>Result: </h3></div>', 
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="text-display-box">{corrected_text}</div>', 
                    unsafe_allow_html=True
                )
        
        # --- Tab 2: Personality Insights (Graphology) ---
        with tab2:
            st.markdown(
                '<div class="tab-subheader"><h3>Graphological Personality Profile</h3></div>', 
                unsafe_allow_html=True
            )
            
            # Run the graphology pipeline
            gray_img, bin_img, lines = preprocess_for_graphology(img_array)
            features = extract_graphological_features(gray_img, bin_img, lines)
            q_profile, d_profile = get_personality_profile(features)
            
            # Create two columns for the charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("##### Trait Profile Radar")
                fig = create_spider_chart(q_profile)
                st.pyplot(fig)
            
            with chart_col2:
                st.markdown("##### Trait Scores (0.1 to 1.0)")
                chart_data = pd.DataFrame(
                    list(q_profile.values()),
                    index=list(q_profile.keys()),
                    columns=["Score"]
                )
                st.bar_chart(chart_data)
            
            st.markdown("---")
            st.markdown("##### Descriptive Analysis")
            for trait, description in d_profile.items():
                st.info(f"**{trait}:** {description}")

        # --- Tab 3: Raw Extracted Features ---
        with tab3:
            st.markdown(
                '<div class="tab-subheader"><h3>Raw Graphological Features</h3></div>', 
                unsafe_allow_html=True
            )
            st.write("These are the raw values extracted from the image and used to generate the personality profile.")
            st.json(features)