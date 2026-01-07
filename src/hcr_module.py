# src/hcr_module.py

import streamlit as st
import pytesseract
from PIL import Image
from spellchecker import SpellChecker
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# --- TESSERACT (PRINT) FUNCTIONS ---

def get_tesseract_transcription(image_array):
    """
    Takes a preprocessed image array and uses Tesseract for print.
    """
    image = Image.fromarray(image_array)
    custom_config = r'--oem 3 --psm 11'
    transcribed_text = pytesseract.image_to_string(image, config=custom_config)
    return transcribed_text

# --- TrOCR (CURSIVE) FUNCTIONS ---

@st.cache_resource
def load_trocr_model():
    """Loads the TrOCR model and processor from Hugging Face."""
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return processor, model

def get_trocr_transcription(processor, model, image_array):
    """
    Takes a TrOCR model and a RAW image array, and returns the transcribed text.
    """
    # TrOCR works best with the original, unprocessed image
    image = Image.fromarray(image_array).convert("RGB")
    
    # The processor prepares the image for the model
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    # The model generates the token IDs, we give it a max length
    generated_ids = model.generate(pixel_values, max_length=256)
    
    # The processor decodes the token IDs back into a human-readable string
    transcribed_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return transcribed_text

# --- SHARED POST-PROCESSING FUNCTION ---

def post_process_text(text):
    """
    Corrects common spelling errors in the transcribed text.
    """
    spell = SpellChecker()
    words = text.split()
    misspelled = spell.unknown(words)
    corrected_text = []
    for word in words:
        if word in misspelled:
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word if corrected_word else word)
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)