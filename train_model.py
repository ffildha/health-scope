import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import re
import numpy as np
from scipy.sparse import hstack

# --- TEXT NORMALIZATION LAYER ---
symptom_map = {
    # Migraine & Hypertension Normalizations
    "severe headache": "headache",
    "head pain": "headache",
    "my head hurts": "headache",
    "തല വേദന": "തലവേദന",

    # Gastritis
    "stomach burning": "burning stomach",
    "burning stomach": "burning stomach",
    "വയറു കത്തുന്നു": "വയറ് കത്തൽ",

    # Asthma
    "breathing problem": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "difficulty in breathing": "shortness of breath",
    "cant breathe": "shortness of breath",
    "ശ്വാസം എടുക്കാൻ ബുദ്ധിമുട്ട്": "ശ്വാസം മുട്ടൽ",

    # Arthritis
    "knee pain": "joint pain",
    "joint stiffness": "stiffness",
    "മുട്ട് വേദന": "മുട്ടുവേദന",
    "my foot hurts": "joint pain",
    "foot pain": "joint pain",
    "pain in foot": "joint pain",
    "leg pain": "joint pain",
    "foot hurting": "joint pain",
    "swelling joints": "swelling",

    # Hypertension
    "feeling dizzy": "dizziness",
    "തല ചുറ്റുന്നു": "തല ചുറ്റൽ"
}

def normalize_text(text):
    text = str(text).lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Replace known symptom variations using the dictionary (sort by length to match longest first)
    sorted_variants = sorted(symptom_map.items(), key=lambda x: len(x[0]), reverse=True)
    for variant, standard in sorted_variants:
        text = text.replace(variant.lower(), standard.lower())
    
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_symptom_weights(text):
    text = str(text).lower()
    weights = {
        "Migraine": 0, "Hypertension": 0, "Diabetes": 0,
        "Asthma": 0, "Gastritis": 0, "Arthritis": 0
    }
    
    # Migraine
    if "headache" in text or "തലവേദന" in text: weights["Migraine"] += 3
    if "nausea" in text or "ഛർദ്ദി" in text: weights["Migraine"] += 2
    if "light sensitivity" in text or "വെളിച്ചം സഹിക്കാത്തത്" in text: weights["Migraine"] += 2
    if "vomiting" in text: weights["Migraine"] += 2

    # Hypertension
    if "dizziness" in text or "തല ചുറ്റൽ" in text: weights["Hypertension"] += 3
    if "blurred vision" in text or "കാഴ്ച മങ്ങൽ" in text: weights["Hypertension"] += 2
    if "chest pressure" in text: weights["Hypertension"] += 2
    if "fatigue" in text or "ക്ഷീണം" in text: weights["Hypertension"] += 1

    # Diabetes
    if "frequent urination" in text or "മൂത്രം കൂടുതലായി പോകുന്നു" in text: weights["Diabetes"] += 3
    if "excessive thirst" in text or "ദാഹം" in text or "വളരെ ദാഹം" in text: weights["Diabetes"] += 3
    if "fatigue" in text or "ക്ഷീണം" in text: weights["Diabetes"] += 2
    if "blurred vision" in text or "കാഴ്ച മങ്ങൽ" in text: weights["Diabetes"] += 2

    # Asthma
    if "shortness of breath" in text or "ശ്വാസം മുട്ടൽ" in text: weights["Asthma"] += 3
    if "wheezing" in text or "വീസിംഗ്" in text: weights["Asthma"] += 3
    if "chest tightness" in text or "നെഞ്ച് കുരുക്ക്" in text: weights["Asthma"] += 2
    if "coughing" in text: weights["Asthma"] += 2

    # Gastritis
    if "stomach burning" in text or "വയറു കത്തൽ" in text or "വയറ് കത്തൽ" in text: weights["Gastritis"] += 3
    if "acidity" in text or "അമ്ലം" in text: weights["Gastritis"] += 3
    if "bloating" in text: weights["Gastritis"] += 2
    if "nausea" in text or "ഛർദ്ദി" in text: weights["Gastritis"] += 2

    # Arthritis
    if "joint pain" in text or "സന്ധിവേദന" in text or "മുട്ടുവേദന" in text or "knee pain" in text: weights["Arthritis"] += 3
    if "stiffness" in text or "സന്ധി മുറുകൽ" in text: weights["Arthritis"] += 2
    if "swelling" in text or "വീക്കം" in text: weights["Arthritis"] += 2

    return [
        weights["Migraine"], weights["Hypertension"], weights["Diabetes"],
        weights["Asthma"], weights["Gastritis"], weights["Arthritis"]
    ]

def train_model():
    print("Loading 3000-row informal dataset...")
    # Load the bilingual dataset
    df = pd.read_csv('symptoms_dataset.csv')
    
    # 1. Preprocess Text
    # Apply normalization and remove heavy punctuation
    df['symptoms'] = df['symptoms'].apply(normalize_text)
    
    print("Vectorizing Text Data (TF-IDF with Word N-grams)...")
    # Convert symptom text into TF-IDF numerical vectors using word n-grams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, lowercase=True)
    X_tfidf = vectorizer.fit_transform(df['symptoms'])
    
    print("Calculating Symptom Weights...")
    # Calculate symptom weights for all rows
    weight_features = np.array(df['symptoms'].apply(get_symptom_weights).tolist())
    
    # Concatenate TF-IDF features with Symptom Weights
    X = hstack([X_tfidf, weight_features])
    y = df['disease']
    
    # 2. Split Data for Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Logistic Regression Model on 80% split...")
    # Initialize and train the classifier
    # We use logistic regression because we need accurate .predict_proba() arrays for confidence scores
    eval_model = LogisticRegression(random_state=42, max_iter=1000)
    eval_model.fit(X_train, y_train)
    
    # Evaluate Accuracy
    y_pred = eval_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Evaluation Complete! Accuracy on unseen 20% test set: {:.2f}%".format(accuracy * 100))
    
    # 3. Train Production Model on 100% of data
    print("Training FINAL model on 100% of the dataset for production deployment...")
    final_model = LogisticRegression(random_state=42, max_iter=1000)
    final_model.fit(X, y)
    
    print("Saving pipeline objects...")
    # Save the trained model
    with open('disease_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
        
    # Save the fitted vectorizer
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Successfully generated 'disease_model.pkl' and 'tfidf_vectorizer.pkl' in the root directory!")

if __name__ == '__main__':
    train_model()
