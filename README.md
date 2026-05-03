# REVE AI - Advanced Dermatology Intelligence

REVE AI is an advanced, end-to-end platform for skin disease and cancer prediction. Designed specifically for dermatologists and medical professionals, it utilizes deep learning models alongside Agentic AI to deliver highly accurate classifications, confidence scores, and comprehensive medical reports.

## System Architecture

The application has recently been upgraded from a basic Streamlit UI to a robust, premium web architecture:
- **Backend:** FastAPI server (`main.py`) running locally.
- **Frontend:** Premium, glassmorphism-styled JavaScript, HTML, and Vanilla CSS UI.
- **Database:** A lightweight local SQLite database (`database.db`) persists uploaded images, model predictions, and generated reports.
- **AI Core:** Integrated with custom-trained PyTorch CNN models (`predict_c.py`, `predict_d.py`) and Google Gemini (via `agno`) for generating medical reports and verification.

## Features

- **Upload & Capture:** Upload skin images directly from your device or capture them using a webcam.
- **Agentic AI Verification:** Employs advanced LLMs to verify if an image is healthy or unhealthy before deep analysis.
- **Deep Learning Predictions:** Uses local CNNs to diagnose specific skin conditions (e.g., Melanoma, Basal Cell Carcinoma).
- **Persistent Records:** All analyses are stored locally in a SQLite database along with the image inside an `uploads/` directory for historical tracking.
- **Responsive Premium UI:** A dynamic and beautiful interface to improve the clinical user experience.

## Quickstart

### 1) Prerequisites

Ensure you have Python 3.9+ installed.

Activate your virtual environment (if using one):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install Dependencies

Install the necessary libraries for the backend and AI models:
```bash
pip install fastapi uvicorn[standard] python-multipart pydantic requests
pip install torch torchvision torchaudio
pip install pillow opencv-python sqlite3 agno google-genai timm
```

### 3) Run the Application

Start the FastAPI server:
```bash
python main.py
```

The server will automatically generate the `static/` folder logic, the `uploads/` folder, and initialize `database.db`.

Once running, navigate to the web interface in your browser:
**[http://127.0.0.1:6701](http://127.0.0.1:6701)**

### 4) View Stored Records (Database)

All prediction data is inserted into the `database.db` SQLite file under the `records` table. You can use any standard SQLite viewer to read historical logs, and the images are safely stored in the `uploads/` directory.

## Disclaimer

⚠️ **For Professional Medical Use Only.** 
This AI-generated diagnosis report should not replace professional medical consultation. Always consult a certified dermatologist for a confirmed diagnosis and personalized treatment.
