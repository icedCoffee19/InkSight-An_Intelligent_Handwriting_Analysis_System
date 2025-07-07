# 🖋️ InkSight

**InkSight** is a modular, AI-powered web application that transforms handwritten input into rich insights. It features:

- ✍️ **Handwritten Character Recognition (HCR)** – Converts scanned or photographed handwriting into digital text.
- 🧠 **Graphology-based Personality Profiling** – Analyzes handwriting traits to infer psychological characteristics such as personality type, mood, stress, and cognitive tendencies.
- 📊 **Interactive Dashboard** – Presents results with clear visualizations, enabling interpretation of both textual and behavioral outputs.

## 🔍 Features

- Upload handwriting images from any device.
- Choose between:
  - Handwriting Transcription (HCR)
  - Personality Prediction (Graphology)
  - Or both
- Visualize results using BI dashboards.
- Built with beginner-friendly yet scalable technologies (React.js, FastAPI, Python, TensorFlow/PyTorch).

## 📁 Project Structure (Highlights)
InkSight/
├── frontend/ # React frontend for user interaction
├── backend/ # FastAPI backend for ML processing
├── data/ # Raw and processed handwriting datasets
├── models/ # Pretrained and custom trained models
├── docs/ # Diagrams and documentation

## 🚀 Getting Started
### Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

### Frontend
cd frontend
npm install
npm start

📚 License
MIT License

InkSight is currently under active development !!