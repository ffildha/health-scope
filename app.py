from flask import Flask, render_template, request, redirect, url_for, session, flash
from datetime import datetime
import sqlite3
import re
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
from scipy.sparse import hstack


app = Flask(__name__)

# --- LOAD ML MODEL GLOBALLY ---
try:
    with open('disease_model.pkl', 'rb') as f:
        ml_model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        ml_vectorizer = pickle.load(f)
except FileNotFoundError:
    print("WARNING: ML Models not found! Run 'python train_model.py' first.")
    ml_model = None
    ml_vectorizer = None
# Secret key is needed for session management and flash messages
import os
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key-for-local-dev') 

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Create the users table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    # cursor.execute('DROP TABLE IF EXISTS history')  # REMOVED: To persist user history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symptoms TEXT,
            predicted_disease TEXT,
            confidence INTEGER,
            department TEXT,
            language TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Admin table initialization
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    
    # Initialize a default admin if none exists (healthscope@gmail.com / health123)
    cursor.execute('SELECT * FROM admins WHERE email = ?', ('healthscope@gmail.com',))
    if not cursor.fetchone():
        hashed_pw = generate_password_hash('health123')
        cursor.execute('INSERT INTO admins (name, email, password) VALUES (?, ?, ?)', 
                       ('Main Admin', 'healthscope@gmail.com', hashed_pw))

    conn.commit()
    conn.close()

# Helper function to get database connection
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row # This allows column access by name (e.g., user['email'])
    return conn




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
    "cannot breathe": "shortness of breath",
    "cant breath": "shortness of breath",
    "run out of breath": "shortness of breath",
    "run out of breth": "shortness of breath",
    "breth": "breath",
    "breath easily": "shortness of breath",
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
    # Replace known symptom variations using the dictionary
    sorted_variants = sorted(symptom_map.items(), key=lambda x: len(x[0]), reverse=True)
    for variant, standard in sorted_variants:
        text = text.replace(variant.lower(), standard.lower())
    return re.sub(r'\s+', ' ', text).strip()

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
        # Weight increase: +1 if > 3 days, +2 if > 7 days
        if days > 7: score = 2
        elif days > 3: score = 1
    
    # 2. Check for Weeks
    week_match = re.search(r'(\d+|one|two|three|four|ഒരു|രണ്ട്|മൂന്ന്|നാല്)\s+(week|weeks|ആഴ്ച|ആഴ്ചയായി)', text)
    if week_match:
        score = 3 # Stronger indicator for chronic symptoms
        
    # 3. Check for Months
    month_match = re.search(r'(\d+|one|two|three|ഒരു|രണ്ട്|മൂന്ന്)\s+(month|months|മാസം|മാസമായി)', text)
    if month_match:
        score = 5 # Very strong chronic indicator
        
    return score

def get_symptom_weights(text):
    text = str(text).lower()
    duration_bonus = get_duration_score(text)
    
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
    if "shortness of breath" in text or "ശ്വാസം മുട്ടൽ" in text or "breath" in text: weights["Asthma"] += 4
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

    # APPLY DURATION BONUS
    # If the user mentioned a long duration, slightly boost all matched categories
    for disease in weights:
        if weights[disease] > 0:
            weights[disease] += duration_bonus

    return [
        weights["Migraine"], weights["Hypertension"], weights["Diabetes"],
        weights["Asthma"], weights["Gastritis"], weights["Arthritis"]
    ]

# Department Mapping
department_mapping = {
    "Migraine": "Neurology",
    "Hypertension": "Cardiology",
    "Diabetes": "Endocrinology",
    "Asthma": "Pulmonology",
    "Gastritis": "Gastroenterology",
    "Arthritis": "Orthopedics"
}

disease_ml_map = {
    "Migraine": "മൈഗ്രൈൻ",
    "Hypertension": "രക്തസമ്മർദ്ദം",
    "Diabetes": "പ്രമേഹം",
    "Asthma": "ആസ്ത്മ",
    "Gastritis": "ഗാസ്ട്രൈറ്റിസ്",
    "Arthritis": "ആർത്ത്രൈറ്റിസ്"
}

department_mapping_ml = {
    "Diabetes": "എൻഡോക്രൈനോളജി വിഭാഗം",
    "Arthritis": "ഓർത്തോപീഡിക്സ് വിഭാഗം",
    "Asthma": "പൾമനോളജി വിഭാഗം",
    "Migraine": "ന്യൂറോളജി വിഭാഗം",
    "Gastritis": "ഗ്യാസ്ട്രോഎന്ററോളജി വിഭാഗം",
    "Hypertension": "കാർഡിയോളജി വിഭാഗം"
}

# Advanced Disease Matching
def match_diseases(raw_text):
    results = {}
    
    # 1. Base Machine Learning Probability scoring
    if ml_model and ml_vectorizer:
        normalized_text = normalize_text(raw_text)
        vectorized_text = ml_vectorizer.transform([normalized_text])
        
        # CRITICAL FIX: Check if the input contains ANY known symptoms from vocabulary or weights
        # If vectorized_text has no non-zero elements AND manual weights are zero, input is meaningless
        symptom_weights = get_symptom_weights(normalized_text)
        if vectorized_text.nnz == 0 and sum(symptom_weights) == 0:
            return {} # No matching disease found
            
        weight_features = np.array([symptom_weights])
        combined_features = hstack([vectorized_text, weight_features])
        
        probabilities = ml_model.predict_proba(combined_features)[0]
        classes = ml_model.classes_
        
        # Extract strict base_score probabilities as requested
        for idx, disease in enumerate(classes):
            prob = float(probabilities[idx])
            confidence = round(prob * 100.0, 1)
            
            # Only keep results where there's a tangible probability to avoid massive clutter
            if confidence > 0.1:
                results[disease] = {
                    "matched": 1, 
                    "total": 1, 
                    "base_score": confidence,
                    "duration_weight": 0,
                    "severity_weight": 0,
                    "confidence": confidence
                }
                
    # Filter out all results if the highest confidence is too low (below 40.0%) 
    # to avoid returning random/low-confidence predictions.
    if results:
        max_conf = max(data['confidence'] for data in results.values())
        if max_conf < 40.0:
            return {}
            
    # Sort diseases by confidence descending
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]['confidence'], reverse=True))
    return sorted_results

@app.route('/')
def home():
    # Show navigation landing page
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Backend validation for name (only alphabets and spaces)
        if not re.match(r'^[A-Za-z\s]+$', name):
            flash('Name should contain only letters (A–Z, a–z)', 'danger')
            return render_template('signup.html')

        # Hash the password for security (never store plain text passwords!)
        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        try:
            # Insert the new user into the database
            conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                         (name, email, hashed_password))
            conn.commit()
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            # Handle the case where the email is already registered
            flash('Email address already exists.', 'danger')
        finally:
            conn.close()

    # If it's a GET request, just show the signup form
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        # Check if user exists and password is correct
        if user and check_password_hash(user['password'], password):
            # Store user data in the session
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            return redirect(url_for('language'))
        else:
            flash('Invalid email or password.', 'danger')

    # If it's a GET request, just show the login form
    return render_template('login.html')

@app.route('/dashboard_en')
def dashboard_en():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    return render_template('dashboard_en.html', name=session['user_name'])

@app.route('/dashboard_ml')
def dashboard_ml():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    return render_template('dashboard_ml.html', name=session['user_name'])

@app.route('/logout')
def logout():
    # Clear the session data to log out the user
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/language')
def language():
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
    return render_template('language.html')

@app.route('/set_language/<lang>')
def set_language(lang):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if lang in ['en', 'ml']:
        session['language'] = lang
        if lang == 'en':
            return redirect(url_for('dashboard_en'))
        elif lang == 'ml':
            return redirect(url_for('dashboard_ml'))
            
    return redirect(url_for('language'))

@app.route('/symptom_en', methods=['GET', 'POST'])
def symptom_en():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        raw_text = request.form['symptoms']
        
        session['symptoms'] = raw_text
        
        matches = match_diseases(raw_text)
        session['matches_en'] = matches

        # Save to History DB
        if matches:
            top_disease = list(matches.keys())[0]
            confidence = int(matches[top_disease]['confidence']) # Store as INTEGER per spec
            department = department_mapping.get(top_disease, "General Physician")
            
            # GENERATE ACCURATE TIMESTAMP IN PYTHON PER SPEC
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            conn = get_db_connection()
            conn.execute('''
                INSERT INTO history (user_id, symptoms, predicted_disease, confidence, department, language, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session['user_id'], raw_text, top_disease, confidence, department, 'en', current_time))
            conn.commit()
            conn.close()
            
        return redirect(url_for('result_en'))
        
    return render_template('symptom_en.html', name=session['user_name'])

@app.route('/symptom_ml', methods=['GET', 'POST'])
def symptom_ml():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        raw_text = request.form['symptoms']
        
        session['symptoms'] = raw_text
        
        matches = match_diseases(raw_text)
        session['matches_ml'] = matches
        
        # Save to History DB
        if matches:
            top_disease = list(matches.keys())[0]
            confidence = int(matches[top_disease]['confidence']) # Store as INTEGER per spec
            department = department_mapping_ml.get(top_disease, "ജനറൽ ഫിസിഷ്യനെ സമീപിക്കുക")
            
            # GENERATE ACCURATE TIMESTAMP IN PYTHON PER SPEC
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            conn = get_db_connection()
            conn.execute('''
                INSERT INTO history (user_id, symptoms, predicted_disease, confidence, department, language, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session['user_id'], raw_text, top_disease, confidence, department, 'ml', current_time))
            conn.commit()
            conn.close()

        return redirect(url_for('result_ml'))
        
    return render_template('symptom_ml.html', name=session['user_name'])

@app.route('/result_en')
def result_en():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    symptoms = session.get('symptoms', '')
    matches = session.get('matches_en', {})
    
    top_disease = None
    top_dept = None
    if matches:
        matches = dict(sorted(matches.items(), key=lambda item: item[1]['confidence'], reverse=True))
        top_disease = list(matches.keys())[0]
        top_dept = department_mapping.get(top_disease, "General Physician")
        
    return render_template('result_en.html', name=session['user_name'], symptoms=symptoms, matches=matches, dept_map=department_mapping)

@app.route('/result_ml')
def result_ml():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    symptoms = session.get('symptoms', '')
    matches = session.get('matches_ml', {})
    top_disease = None
    top_dept = None
    if matches:
        matches = dict(sorted(matches.items(), key=lambda item: item[1]['confidence'], reverse=True))
        top_disease = list(matches.keys())[0]
        top_dept = department_mapping_ml.get(top_disease, "ജനറൽ ഫിസിഷ്യനെ സമീപിക്കുക")
        
    return render_template('result_ml.html', name=session['user_name'], symptoms=symptoms, matches=matches, dept_map=department_mapping_ml, disease_names=disease_ml_map)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    history_records = conn.execute(
        'SELECT * FROM history WHERE user_id = ? ORDER BY timestamp DESC', 
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('history.html', name=session['user_name'], history=history_records)

@app.route('/delete_history/<int:record_id>')
def delete_history(record_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    conn.execute('DELETE FROM history WHERE id = ? AND user_id = ?', (record_id, session['user_id']))
    conn.commit()
    conn.close()
    flash('History record deleted.', 'success')
    return redirect(url_for('history'))

@app.route('/clear_history')
def clear_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    conn.execute('DELETE FROM history WHERE user_id = ?', (session['user_id'],))
    conn.commit()
    conn.close()
    flash('All history cleared.', 'success')
    return redirect(url_for('history'))


# --- ADMIN PANEL ROUTES ---

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Admin access required.', 'danger')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        admin = conn.execute('SELECT * FROM admins WHERE email = ?', (email,)).fetchone()
        conn.close()

        if admin and check_password_hash(admin['password'], password):
            session['admin_id'] = admin['id']
            session['admin_email'] = admin['email']
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials.', 'danger')

    return render_template('admin_login.html')

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    conn = get_db_connection()
    total_users = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    total_predictions = conn.execute('SELECT COUNT(*) FROM history').fetchone()[0]
    recent_activity = conn.execute('''
        SELECT h.*, u.name as user_name 
        FROM history h 
        JOIN users u ON h.user_id = u.id 
        ORDER BY h.timestamp DESC LIMIT 5
    ''').fetchall()
    conn.close()
    
    return render_template('admin_dashboard.html', 
                          total_users=total_users, 
                          total_predictions=total_predictions, 
                          recent_activity=recent_activity)

@app.route('/admin/users')
@admin_required
def admin_users():
    conn = get_db_connection()
    users_list = conn.execute('SELECT * FROM users ORDER BY name ASC').fetchall()
    conn.close()
    return render_template('admin_users.html', users=users_list)

@app.route('/admin/history')
@admin_required
def admin_history():
    user_id = request.args.get('user_id')
    conn = get_db_connection()
    if user_id:
        history_list = conn.execute('''
            SELECT h.*, u.name as user_name 
            FROM history h 
            JOIN users u ON h.user_id = u.id 
            WHERE h.user_id = ?
            ORDER BY h.timestamp DESC
        ''', (user_id,)).fetchall()
    else:
        history_list = conn.execute('''
            SELECT h.*, u.name as user_name 
            FROM history h 
            JOIN users u ON h.user_id = u.id 
            ORDER BY h.timestamp DESC
        ''').fetchall()
    conn.close()
    return render_template('admin_history.html', history=history_list)

@app.route('/admin/delete_user/<int:uid>')
@admin_required
def admin_delete_user(uid):
    conn = get_db_connection()
    conn.execute('DELETE FROM history WHERE user_id = ?', (uid,))
    conn.execute('DELETE FROM users WHERE id = ?', (uid,))
    conn.commit()
    conn.close()
    flash('User and their history deleted successfully.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/delete_history/<int:hid>')
@admin_required
def admin_delete_history(hid):
    conn = get_db_connection()
    conn.execute('DELETE FROM history WHERE id = ?', (hid,))
    conn.commit()
    conn.close()
    flash('Record deleted.', 'success')
    return redirect(url_for('admin_history'))

@app.route('/admin/admins')
@admin_required
def admin_admins():
    conn = get_db_connection()
    admins_list = conn.execute('SELECT * FROM admins ORDER BY name ASC').fetchall()
    conn.close()
    return render_template('admin_admins.html', admins=admins_list)

@app.route('/admin/add-user', methods=['GET', 'POST'])
@admin_required
def admin_add_user():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if not re.match(r'^[A-Za-z\s]+$', name):
            flash('Name should contain only letters (A–Z, a–z)', 'danger')
            return render_template('admin_add_user.html')

        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                         (name, email, hashed_password))
            conn.commit()
            flash('User created successfully.', 'success')
            return redirect(url_for('admin_users'))
        except sqlite3.IntegrityError:
            flash('Email address already exists.', 'danger')
        finally:
            conn.close()
            
    return render_template('admin_add_user.html')

@app.route('/admin/add-admin', methods=['GET', 'POST'])
@admin_required
def admin_add_admin():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO admins (name, email, password) VALUES (?, ?, ?)',
                         (name, email, hashed_password))
            conn.commit()
            flash('New Admin created successfully.', 'success')
            return redirect(url_for('admin_admins'))
        except sqlite3.IntegrityError:
            flash('Email address already exists.', 'danger')
        finally:
            conn.close()
            
    return render_template('admin_add_admin.html')

@app.route('/admin/delete_admin/<int:aid>')
@admin_required
def admin_delete_admin(aid):
    if aid == session.get('admin_id'):
        flash('You cannot delete yourself!', 'danger')
        return redirect(url_for('admin_admins'))
        
    conn = get_db_connection()
    conn.execute('DELETE FROM admins WHERE id = ?', (aid,))
    conn.commit()
    conn.close()
    flash('Admin removed.', 'success')
    return redirect(url_for('admin_admins'))


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id', None)
    session.pop('admin_email', None)
    flash('Admin logged out.', 'info')
    return redirect(url_for('admin_login'))

# Initialize the database when the module is imported (needed for Gunicorn)
init_db()

if __name__ == '__main__':
    app.run(debug=True)
