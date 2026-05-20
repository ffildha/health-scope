import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import re
import unicodedata
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
    "breathing problems": "shortness of breath",
    "breathing difficulty": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "difficulty in breathing": "shortness of breath",
    "trouble breathing": "shortness of breath",
    "hard to breathe": "shortness of breath",
    "breathlessness": "shortness of breath",
    "breathless": "shortness of breath",
    "cant breathe": "shortness of breath",
    "cannot breathe": "shortness of breath",
    "cant breath": "shortness of breath",
    "out of breath": "shortness of breath",
    "run out of breath": "shortness of breath",
    "run out of breth": "shortness of breath",
    "breth": "shortness of breath",
    "gasping for air": "shortness of breath",
    "breath easily": "shortness of breath",
    "wheeze": "wheezing",
    "tight chest": "chest tightness",
    "chest feels tight": "chest tightness",
    "persistent cough": "coughing",
    "dry cough": "coughing",
    "night cough": "coughing",
    "asthmatic": "asthma",
    "asthma attack": "asthma",
    "ആസ്ത്മ": "asthma",
    "ശ്വാസം എടുക്കാൻ ബുദ്ധിമുട്ട്": "ശ്വാസം മുട്ടൽ",
    "ശ്വാസം വലിക്കാൻ ബുദ്ധിമുട്ട്": "ശ്വാസം മുട്ടൽ",
    "ശ്വാസം എടുക്കാൻ പറ്റുന്നില്ല": "ശ്വാസം മുട്ടൽ",
    "ശ്വാസം കിട്ടുന്നില്ല": "ശ്വാസം മുട്ടൽ",
    "ശ്വാസം മുട്ടുന്നു": "ശ്വാസം മുട്ടൽ",
    "ശ്വാസക്കുറവ്": "ശ്വാസം മുട്ടൽ",
    "ശ്വാസതടസം": "ശ്വാസം മുട്ടൽ",
    "കിതപ്പ്": "ശ്വാസം മുട്ടൽ",
    "കിതപ്പുണ്ട്": "ശ്വാസം മുട്ടൽ",
    "വീസിങ്": "വീസിംഗ്",
    "നെഞ്ച് മുറുക്ക്": "നെഞ്ച് കുരുക്ക്",
    "നെഞ്ച് ഇറുക്ക്": "നെഞ്ച് കുരുക്ക്",
    "ചുമ": "coughing",
    "ചുമയുണ്ട്": "coughing",
    "ചുമയ്ക്കുന്നു": "coughing",

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
    "chestpain": "chest pressure",
    "chest pain": "chest pressure",
    "pain in chest": "chest pressure",
    "നെഞ്ച് വേദന": "chest pressure",
    "നെഞ്ചുവേദന": "chest pressure",
    "മാർ വേദന": "chest pressure",
    "മാർവേദന": "chest pressure",
    "feeling dizzy": "dizziness",
    "തല ചുറ്റുന്നു": "തല ചുറ്റൽ",
    "tierd": "fatigue",
    "tired": "fatigue",
    "feeling tired": "fatigue",
    "very tired": "fatigue",
    "dehydration": "dehydrated",
    "dehydrated": "dehydrated",
    "thirsty": "excessive thirst",
    "very thirsty": "excessive thirst",
    "feeling thirsty": "excessive thirst",
    "dry mouth": "excessive thirst",
    "തളർച്ച": "ക്ഷീണം",
    "തളർച്ചയുണ്ട്": "ക്ഷീണം",
    "തളർന്നിരിക്കുന്നു": "ക്ഷീണം",
    "ദാഹിക്കുന്നു": "ദാഹം",
    "വളരെ ദാഹിക്കുന്നു": "ദാഹം",
    "വായ് വരൾച്ച": "ദാഹം",
    "നിർജ്ജലീകരണം": "dehydrated",
    "നിർജലീകരണം": "dehydrated"
}

ASTHMA_DISEASE_TERMS = ["asthma"]
ASTHMA_BREATHING_TERMS = ["shortness of breath", "ശ്വാസം മുട്ടൽ"]
ASTHMA_WHEEZING_TERMS = ["wheezing", "വീസിംഗ്"]
ASTHMA_CHEST_TIGHTNESS_TERMS = ["chest tightness", "നെഞ്ച് കുരുക്ക്"]
ASTHMA_COUGH_TERMS = ["coughing"]
HYPERTENSION_CHEST_TERMS = ["chest pressure"]
HYPERTENSION_FATIGUE_TERMS = ["fatigue", "ക്ഷീണം"]
DIABETES_THIRST_TERMS = ["excessive thirst", "ദാഹം"]
DIABETES_DEHYDRATION_TERMS = ["dehydrated"]
DIABETES_FATIGUE_TERMS = ["fatigue", "ക്ഷീണം", "tired and dehydrated", "ക്ഷീണവും നിർജ്ജലീകരണവും"]

def remove_punctuation(text):
    return ''.join(char for char in text if not unicodedata.category(char).startswith('P'))

def normalize_text(text):
    text = str(text).lower()
    # Remove punctuation
    text = remove_punctuation(text)
    
    # Replace known symptom variations using the dictionary (sort by length to match longest first)
    sorted_variants = sorted(symptom_map.items(), key=lambda x: len(x[0]), reverse=True)
    for variant, standard in sorted_variants:
        text = text.replace(variant.lower(), standard.lower())
    
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- DURATION DETECTION LAYER ---
def get_duration_score(text):
    text = str(text).lower()
    score = 0
    
    # Mapping written numbers to numerical values (EN & ML)
    num_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "ഒരു": 1, "രണ്ട്": 2, "മൂന്ന്": 3, "നാല്": 4, "അഞ്ച്": 5, "ആറ്": 6, "ഏഴ്": 7,
        "one month": 30, "two months": 60, "three months": 90, "ഒരു മാസം": 30
    }

    # 1. Check for Days
    day_match = re.search(r'(\d+|one|two|three|four|five|six|seven|ഒരു|രണ്ട്|മൂന്ന്|നാല്|അഞ്ച്|ആറ്|ഏഴ്)\s+(day|days|ദിവസം|ദിവസമായി)', text)
    if day_match:
        val = day_match.group(1)
        days = int(val) if val.isdigit() else num_map.get(val, 0)
        if days > 7: score = 2
        elif days > 3: score = 1
    
    # 2. Check for Weeks
    week_match = re.search(r'(\d+|one|two|three|four|ഒരു|രണ്ട്|മൂന്ന്|നാല്)\s+(week|weeks|ആഴ്ച|ആഴ്ചയായി)', text)
    if week_match:
        score = 3
        
    # 3. Check for Months
    month_match = re.search(r'(\d+|one|two|three|ഒരു|രണ്ട്|മൂന്ന്)\s+(month|months|മാസം|മാസമായി)', text)
    if month_match:
        score = 5
        
    return score

def get_symptom_weights(text):
    text = normalize_text(text)
    duration_bonus = get_duration_score(text)
    
    weights = {
        "Migraine": 0, "Hypertension": 0, "Diabetes": 0,
        "Asthma": 0, "Gastritis": 0, "Arthritis": 0
    }
    
    # Migraine
    if any(word in text for word in ["headache", "head ache", "severe headache", "തലവേദന", "തല വേദന", "തലവേദനയുണ്ട്","തലവേദനയാണ്", "കടുത്ത തലവേദന", "ഭയങ്കര തലവേദന"]): weights["Migraine"] += 3
    if any(word in text for word in ["nausea", "ഓക്കാനം", "ഛർദ്ദിക്കാൻ വരുന്നു"]): weights["Migraine"] += 2
    if any(word in text for word in ["light sensitivity", "വെളിച്ചം സഹിക്കാത്തത്", "വെളിച്ചം കാണാൻ വയ്യ", "കണ്ണിലേക്ക് വെളിച്ചം അടിക്കാൻ വയ്യ"]): weights["Migraine"] += 2
    if any(word in text for word in ["vomiting", "ഛർദ്ദി"]): weights["Migraine"] += 2

    # Hypertension
    if any(word in text for word in ["dizziness", "dizzy", "feeling dizzy", "തല ചുറ്റൽ", "തലചുറ്റൽ", "തല ചുറ്റുന്നു", "തല കറക്കം", "തല കറങ്ങുന്നു"]): weights["Hypertension"] += 3
    if any(word in text for word in ["blurred vision", "blurry vision", "bluzzed vision", "കാഴ്ച മങ്ങൽ", "കാഴ്ച മങ്ങുന്നു", "കണ്ണിന് മങ്ങൽ", "കാഴ്ചയ്ക്ക് മങ്ങൽ"]): weights["Hypertension"] += 2
    if any(word in text for word in HYPERTENSION_CHEST_TERMS): weights["Hypertension"] += 2
    if any(word in text for word in HYPERTENSION_FATIGUE_TERMS): weights["Hypertension"] += 1

    # Diabetes
    if "frequent urination" in text or "മൂത്രം കൂടുതലായി പോകുന്നു" in text: weights["Diabetes"] += 3
    if "excessive thurst" in text or any(word in text for word in DIABETES_THIRST_TERMS): weights["Diabetes"] += 3
    if any(word in text for word in DIABETES_DEHYDRATION_TERMS): weights["Diabetes"] += 2
    if any(word in text for word in DIABETES_FATIGUE_TERMS): weights["Diabetes"] += 2
    if "blurred vision" in text or "കാഴ്ച മങ്ങൽ" in text: weights["Diabetes"] += 2

    # Asthma
    if any(word in text for word in ASTHMA_DISEASE_TERMS): weights["Asthma"] += 5
    if any(word in text for word in ASTHMA_BREATHING_TERMS): weights["Asthma"] += 4
    if any(word in text for word in ASTHMA_WHEEZING_TERMS): weights["Asthma"] += 3
    if any(word in text for word in ASTHMA_CHEST_TIGHTNESS_TERMS): weights["Asthma"] += 2
    if any(word in text for word in ASTHMA_COUGH_TERMS): weights["Asthma"] += 2

    # Gastritis
    if "stomach burning" in text or "വയറു കത്തൽ" in text or "വയറ് കത്തൽ" in text: weights["Gastritis"] += 3
    if "acidity" in text or "അമ്ലം" in text: weights["Gastritis"] += 3
    if "bloating" in text: weights["Gastritis"] += 2
    if "nausea" in text or "ഛർദ്ദി" in text: weights["Gastritis"] += 2

    # Arthritis
    if "joint pain" in text or "സന്ധിവേദന" in text or "മുട്ടുവേദന" in text or "knee pain" in text: weights["Arthritis"] += 3
    if "stiffness" in text or "സന്ധി മുറുകൽ" in text: weights["Arthritis"] += 2
    if "swelling" in text or "വീക്കം" in text: weights["Arthritis"] += 2

    # APPLY DURATION BONUS
    for disease in weights:
        if weights[disease] > 0:
            weights[disease] += duration_bonus

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
