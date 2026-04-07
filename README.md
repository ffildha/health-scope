# HEALTH-SCOPE: Bilingual Disease Prediction System 🏥

HEALTH-SCOPE is a Flask-powered web application that uses machine learning to predict potential diseases based on user symptoms. It supports both **English** and **Malayalam**, featuring high-accuracy prediction logic and a secure user history system with voice-enabled input.

## 🚀 Features

- **Bilingual Interface**: Seamlessly switch between English and Malayalam for symptom entry and results.
- **Bilingual Voice Input**:
  - Integrated **Web Speech API** for hands-free symptom entry.
  - Supports both **English (en-US)** and **Malayalam (ml-IN)** voice recognition.
  - Interactive "Listening..." feedback and appending support for multi-turn entry.
- **Advanced Prediction Reliability**:
  - **40% Confidence Threshold**: Rejects low-confidence predictions to ensure accuracy.
  - **Input Pre-Validation**: Detects and rejects meaningless or random text inputs (e.g., "jfbdjd") that don't match medical vocabulary.
- **Secure Authentication**:
  - User registration and login with encrypted passwords.
  - **Name Validation**: Accepts only alphabetic characters (A-Z, a-z) for accurate user data.
- **Persistent User History**:
  - Tracks diagnostic history securely per user.
  - **Accurate Timestamping**: Real-time Python-generated timestamps for each prediction.
- **History Management**:
  - Options to **delete individual records** or **clear all history**.
  - Optimized table layout with horizontal scrolling for consistent mobile usage.
- **Machine Learning Analysis**: Uses a Logistic Regression model trained on 3,000+ symptom records.
- **Hybrid Scoring**: Combines TF-IDF vectors with custom symptom weighting for high precision.

## 📋 System Architecture

1.  **Frontend**: HTML5, CSS3 (Vanilla), and JavaScript (Web Speech API).
2.  **Backend**: Python Flask (Session management & result routing).
3.  **Database**: SQLite (Persistent user management and history).
4.  **AI Engine**: Scikit-Learn (Logistic Regression, TF-IDF Vectorization).

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/ffildha/health-scope.git
    cd health-scope
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model** (Required for first run):
    ```bash
    python train_model.py
    ```

4.  **Start the Application**:
    ```bash
    python app.py
    ```
    Access the app at `http://127.0.0.1:5000/`.

## 📁 Project Structure

- `app.py`: The main Flask application (routing, DB, result matching).
- `train_model.py`: Training script for the ML pipeline.
- `generate_dataset.py`: Script to generate the synthetic bilingual dataset.
- `templates/`: HTML templates for auth, dashboard, and result views.
- `users.db`: SQLite database for user data and persistent history (automatically initialized).
- `*.pkl`: Serialized machine learning models and TF-IDF vectorizers.

## ⚖️ Disclaimer

*This application is for educational purposes only. The results provided are based on statistical patterns and should NOT be taken as professional medical advice. Always consult a qualified healthcare provider for any health-related concerns.*

---

**Developed with ❤️ by [Fildha](https://github.com/ffildha)**
