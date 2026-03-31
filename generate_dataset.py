import csv
import random

def generate_dataset(filename="symptoms_dataset.csv", rows_per_disease=500):
    data = []

    disease_patterns = {
        "Migraine": [
            ["headache"],
            ["severe headache"],
            ["nausea"],
            ["dizziness"],
            ["light sensitivity"],
            ["തലവേദന"],
            ["severe headache", "nausea"],
            ["headache", "light sensitivity"],
            ["തലവേദന", "ഛർദ്ദി"]
        ],
        "Hypertension": [
            ["high blood pressure", "dizziness"],
            ["blurred vision", "fatigue"],
            ["chest pressure"],
            ["തല ചുറ്റൽ"],
            ["കാഴ്ച മങ്ങൽ"],
            ["high blood pressure", "fatigue"],
            ["fatigue", "dizziness"]
        ],
        "Diabetes": [
            ["frequent urination", "excessive thirst"],
            ["fatigue", "blurred vision"],
            ["weight loss", "excessive thirst"],
            ["മൂത്രം കൂടുതലായി പോകുന്നു", "ദാഹം"],
            ["ക്ഷീണം", "കാഴ്ച മങ്ങൽ"],
            ["frequent urination", "excessive thirst", "fatigue"]
        ],
        "Asthma": [
            ["shortness of breath"],
            ["difficulty breathing"],
            ["wheezing"],
            ["chest tightness"],
            ["ശ്വാസം മുട്ടൽ"],
            ["shortness of breath", "wheezing"],
            ["difficulty breathing", "chest tightness"]
        ],
        "Gastritis": [
            ["burning stomach"],
            ["acidity"],
            ["bloating"],
            ["stomach burning", "acidity"],
            ["വയറു കത്തൽ"],
            ["അമ്ലം"],
            ["stomach burning", "nausea"]
        ],
        "Arthritis": [
            ["joint pain"],
            ["knee pain"],
            ["stiffness"],
            ["swelling joints"],
            ["മുട്ടുവേദന"],
            ["joint pain", "stiffness"],
            ["knee pain", "swelling joints"],
            ["സന്ധിവേദന", "വീക്കം"]
        ]
    }

    # Add extra random noise symptoms for variability within the disease
    extra_symptoms = {
        "Migraine": ["vomiting", "aura", "one-sided pain", "ഛർദ്ദി", "വെളിച്ചം സഹിക്കാത്തത്"],
        "Hypertension": ["palpitations", "blurred vision", "തലവേദന"],
        "Diabetes": ["weight loss", "ക്ഷീണം", "വളരെ ദാഹം"],
        "Asthma": ["coughing", "ശ്വാസം എടുക്കാൻ ബുദ്ധിമുട്ട്", "നെഞ്ച് കുരുക്ക്"],
        "Gastritis": ["heartburn", "loss of appetite", "gas", "അമ്ലം", "ഛർദ്ദി"],
        "Arthritis": ["joint swelling", "tenderness", "limited movement", "സന്ധി മുറുകൽ"]
    }

    # Generate rows
    for disease, patterns in disease_patterns.items():
        for _ in range(rows_per_disease):
            # Pick a core pattern
            pattern = random.choice(patterns)
            symptoms_list = list(pattern)
            
            # Optionally add 1-2 random extra symptoms from the disease's extra list
            if random.random() > 0.3:
                num_extras = random.randint(1, 2)
                extras = random.sample(extra_symptoms[disease], min(num_extras, len(extra_symptoms[disease])))
                for ext in extras:
                    if ext not in symptoms_list:
                        symptoms_list.append(ext)
            
            # Shuffle the order
            random.shuffle(symptoms_list)
            
            symptoms_str = " ".join(symptoms_list)
            data.append([symptoms_str, disease])

    # Shuffle the entire dataset
    random.shuffle(data)

    # Write to CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["symptoms", "disease"])
        writer.writerows(data)
        
    print(f"Successfully generated {len(data)} rows in '{filename}'!")

if __name__ == "__main__":
    generate_dataset()
