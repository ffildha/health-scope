# HEALTH-SCOPE: Bilingual Disease Prediction System 🏥

HEALTH-SCOPE is a Flask-powered web application that uses machine learning to predict potential diseases based on user symptoms. It supports both **English** and **Malayalam**, featuring an integrated **web speech API** for hands-free voice input.

## 🚀 Features

- **Bilingual Interface**: Seamlessly switch between English and Malayalam.
- **Voice-to-Text Input**: Enter symptoms using your voice via the Web Speech API.
- **Machine Learning Analysis**: Uses a Logistic Regression model trained on 3,000+ informal symptom records.
- **Hybrid Scoring**: Combines TF-IDF vectors with custom symptom weighting for high accuracy.
- **User History**: Securely tracks your previous searches and results using SQLite.
- **Modern UI**: Clean, responsive design for mobile and desktop access.

## 📋 System Architecture

1.  **Frontend**: HTML5, CSS3, & JavaScript (Web Speech API).
2.  **Backend**: Python Flask.
3.  **Database**: SQLite (User management and diagnostic history).
4.  **AI Engine**: Scikit-Learn (Logistic Regression, TF-IDF Vectorization).

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/ffildha/health-scope.git
    cd health-scope
    ```

2.  **Install Dependencies**:
    ```bash
    pip install Flask scikit-learn pandas numpy scipy
    ```

3.  **Prepare the Dataset & Train the Model**:
    ```bash
    python generate_dataset.py
    python train_model.py
    ```

4.  **Start the Application**:
    ```bash
    python app.py
    ```
    Access the app at `http://127.0.0.1:5000/`.

## 📁 Project Structure

- `app.py`: The main Flask application (routing, DB, matching logic).
- `generate_dataset.py`: Script to generate the synthetic bilingual dataset.
- `train_model.py`: Training script for the ML pipeline.
- `templates/`: HTML views for dashboard, results, and settings.
- `users.db`: User and history database (auto-generated).
- `*.pkl`: Serialized machine learning models and vectorizers.

## ⚖️ Disclaimer

*This application is for educational purposes only. The results provided are based on statistical patterns and should NOT be taken as professional medical advice. Always consult a qualified healthcare provider for any health-related concerns.*

---

**Developed with ❤️ by [Fildha](https://github.com/ffildha)**
