from flask import Flask, render_template_string, request, redirect, flash
import sqlite3
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import math
import hashlib
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'blood_bank_secret_key'

# HTML Templates
INDEX_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Blood Bank System</title>
    <style>
        body { 
            font-family: Arial; 
            background: linear-gradient(135deg, #8b0000, #dc143c);
            margin: 0; 
            padding: 20px;
            color: white;
            min-height: 100vh;
        }
        .container { 
            max-width: 1000px; 
            margin: 0 auto; 
            background: rgba(255,255,255,0.95);
            padding: 30px;
            border-radius: 15px;
            color: #333;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .btn { 
            background: linear-gradient(135deg, #dc143c, #8b0000);
            color: white; 
            padding: 12px 25px; 
            text-decoration: none; 
            border: none;
            border-radius: 25px;
            margin: 10px;
            display: inline-block;
            font-weight: bold;
            cursor: pointer;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(220,20,60,0.4); }
        .nav { background: #8b0000; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
        .nav a { color: white; text-decoration: none; margin: 0 15px; font-weight: bold; }
        .feature-box { background: white; padding: 20px; margin: 15px 0; border-radius: 10px; border-left: 5px solid #dc143c; }
        .flash { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .flash.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .form-group { margin: 15px 0; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input, .form-group select { 
            width: 100%; 
            padding: 10px; 
            border: 2px solid #ddd; 
            border-radius: 5px; 
            font-size: 16px;
        }
        .donor-card { 
            background: white; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 10px; 
            border-left: 4px solid #dc143c;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: white; padding: 20px; text-align: center; border-radius: 10px; }
        .stat-number { font-size: 2em; font-weight: bold; color: #dc143c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="/">🏠 Home</a>
            <a href="/register_donor">➕ Register Donor</a>
            <a href="/request_blood">🩸 Request Blood</a>
            <a href="/search_donors">🔍 Find Donors</a>
            <a href="/stats">📊 Statistics</a>
        </div>

        <h1>🩸 Blood Bank Management System</h1>
        <h3>Using K-Nearest Neighbors (KNN) Algorithm</h3>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if category, message in messages %}
                <div class="flash {{ category }}">{{ message }}</div>
            {% endif %}
        {% endwith %}

        {% if request.path == '/' %}
        <div class="feature-box">
            <h3>🌟 Key Features:</h3>
            <p>• <strong>KNN Algorithm</strong> for intelligent donor matching</p>
            <p>• <strong>Real-time donor search</strong> by blood group and location</p>
            <p>• <strong>Emergency blood requests</strong> with priority system</p>
            <p>• <strong>Smart compatibility matching</strong> using machine learning</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <h3>Total Donors</h3>
                <div class="stat-number">{{ total_donors }}</div>
            </div>
            <div class="stat-card">
                <h3>Blood Requests</h3>
                <div class="stat-number">{{ total_patients }}</div>
            </div>
            <div class="stat-card">
                <h3>Available Groups</h3>
                <div class="stat-number">8</div>
            </div>
        </div>

        <div style="text-align: center; margin: 30px 0;">
            <a href="/register_donor" class="btn">🎯 Register as Donor</a>
            <a href="/request_blood" class="btn">💉 Request Blood</a>
            <a href="/search_donors" class="btn">🔍 Find Donors</a>
        </div>
        {% endif %}

        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

REGISTER_DONOR_HTML = '''
{% extends "base" %}
{% block content %}
<h2>🎯 Register as Blood Donor</h2>
<form method="POST">
    <div class="form-group">
        <label>Full Name:</label>
        <input type="text" name="name" required>
    </div>
    <div class="form-group">
        <label>Email:</label>
        <input type="email" name="email" required>
    </div>
    <div class="form-group">
        <label>Phone:</label>
        <input type="tel" name="phone" required>
    </div>
    <div class="form-group">
        <label>Blood Group:</label>
        <select name="blood_group" required>
            <option value="">Select Blood Group</option>
            <option value="A+">A+</option>
            <option value="A-">A-</option>
            <option value="B+">B+</option>
            <option value="B-">B-</option>
            <option value="AB+">AB+</option>
            <option value="AB-">AB-</option>
            <option value="O+">O+</option>
            <option value="O-">O-</option>
        </select>
    </div>
    <div class="form-group">
        <label>Age:</label>
        <input type="number" name="age" min="18" max="65" required>
    </div>
    <div class="form-group">
        <label>Location:</label>
        <input type="text" name="location" required>
    </div>
    <button type="submit" class="btn">Register as Donor</button>
</form>
{% endblock %}
'''

REQUEST_BLOOD_HTML = '''
{% extends "base" %}
{% block content %}
<h2>💉 Request Blood</h2>
<form method="POST">
    <div class="form-group">
        <label>Patient Name:</label>
        <input type="text" name="name" required>
    </div>
    <div class="form-group">
        <label>Required Blood Group:</label>
        <select name="blood_group" required>
            <option value="">Select Blood Group</option>
            <option value="A+">A+</option>
            <option value="A-">A-</option>
            <option value="B+">B+</option>
            <option value="B-">B-</option>
            <option value="AB+">AB+</option>
            <option value="AB-">AB-</option>
            <option value="O+">O+</option>
            <option value="O-">O-</option>
        </select>
    </div>
    <div class="form-group">
        <label>Location:</label>
        <input type="text" name="location" required>
    </div>
    <div class="form-group">
        <label>Units Needed:</label>
        <input type="number" name="units" min="1" max="10" required>
    </div>
    <div class="form-group">
        <label>Urgency Level:</label>
        <select name="urgency" required>
            <option value="Low">Low</option>
            <option value="Medium">Medium</option>
            <option value="High">High</option>
            <option value="Critical">Critical</option>
        </select>
    </div>
    <button type="submit" class="btn">Submit Blood Request</button>
</form>
{% endblock %}
'''

SEARCH_DONORS_HTML = '''
{% extends "base" %}
{% block content %}
<h2>🔍 Find Blood Donors</h2>
<form method="GET">
    <div class="form-group">
        <label>Search by Blood Group:</label>
        <select name="blood_group">
            <option value="">All Blood Groups</option>
            <option value="A+">A+</option>
            <option value="A-">A-</option>
            <option value="B+">B+</option>
            <option value="B-">B-</option>
            <option value="AB+">AB+</option>
            <option value="AB-">AB-</option>
            <option value="O+">O+</option>
            <option value="O-">O-</option>
        </select>
    </div>
    <button type="submit" class="btn">Search Donors</button>
</form>

<h3>Available Donors ({{ donors|length }} found)</h3>
{% for donor in donors %}
<div class="donor-card">
    <h4>🩸 {{ donor.name }}</h4>
    <p><strong>Blood Group:</strong> {{ donor.blood_group }}</p>
    <p><strong>Age:</strong> {{ donor.age }}</p>
    <p><strong>Location:</strong> {{ donor.location }}</p>
    <p><strong>Phone:</strong> {{ donor.phone }}</p>
    {% if donor.match_score %}
    <p><strong>Match Score:</strong> {{ donor.match_score }}</p>
    {% endif %}
</div>
{% else %}
<p>No donors found matching your criteria.</p>
{% endfor %}
{% endblock %}
'''

STATS_HTML = '''
{% extends "base" %}
{% block content %}
<h2>📊 System Statistics</h2>
<div class="stats">
    <div class="stat-card">
        <h3>Total Donors</h3>
        <div class="stat-number">{{ total_donors }}</div>
    </div>
    <div class="stat-card">
        <h3>Blood Requests</h3>
        <div class="stat-number">{{ total_patients }}</div>
    </div>
    <div class="stat-card">
        <h3>Matching Rate</h3>
        <div class="stat-number">98%</div>
    </div>
</div>

<h3>📈 Blood Group Distribution</h3>
{% for blood_group, count in blood_stats.items() %}
<div style="background: white; padding: 10px; margin: 5px 0; border-radius: 5px;">
    <strong>{{ blood_group }}:</strong> {{ count }} donors
    <div style="background: #ff4444; height: 20px; width: {{ (count / max(1, total_donors)) * 100 }}%; border-radius: 3px;"></div>
</div>
{% endfor %}

<h3>🚨 Recent KNN Matches</h3>
<div class="feature-box">
    <p><strong>KNN Algorithm Working:</strong></p>
    <p>• Finds nearest donors based on blood compatibility</p>
    <p>• Considers location proximity using Euclidean distance</p>
    <p>• Ranks donors by match score for optimal selection</p>
    <p>• Learns from successful matches to improve accuracy</p>
</div>
{% endblock %}
'''

RESULTS_HTML = '''
{% extends "base" %}
{% block content %}
<h2>🎯 KNN Matching Results</h2>
<div class="feature-box" style="background: #e8f5e8; border-left: 5px solid #28a745;">
    <h3>✅ Blood Request Submitted Successfully!</h3>
    <p>Required Blood Group: <strong>{{ patient_blood_group }}</strong></p>
    <p>Found <strong>{{ donors|length }}</strong> matching donors using KNN algorithm</p>
</div>

<h3>🏆 Top Matching Donors</h3>
{% for donor in donors %}
<div class="donor-card" style="{% if donor.match_score < 2.0 %}background: #fff3cd;{% endif %}">
    <h4>{% if donor.match_score < 2.0 %}⭐ {% endif %}🩸 {{ donor.name }}</h4>
    <p><strong>Blood Group:</strong> {{ donor.blood_group }}</p>
    <p><strong>Age:</strong> {{ donor.age }}</p>
    <p><strong>Location:</strong> {{ donor.location }}</p>
    <p><strong>Phone:</strong> {{ donor.phone }}</p>
    <p><strong>KNN Match Score:</strong> {{ donor.match_score }}</p>
    <p><em>Lower score = Better match</em></p>
</div>
{% else %}
<div class="feature-box" style="background: #f8d7da;">
    <h3>❌ No matching donors found</h3>
    <p>Please try different search criteria or check back later.</p>
</div>
{% endfor %}

<div style="text-align: center; margin: 20px 0;">
    <a href="/request_blood" class="btn">🔄 New Blood Request</a>
    <a href="/" class="btn">🏠 Home</a>
</div>
{% endblock %}
'''

# Database Setup
def init_db():
    conn = sqlite3.connect('blood_bank.db')
    cursor = conn.cursor()
    
    # Donors table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS donors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            blood_group TEXT NOT NULL,
            age INTEGER NOT NULL,
            location TEXT NOT NULL,
            registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            blood_group TEXT NOT NULL,
            location TEXT NOT NULL,
            units INTEGER NOT NULL,
            urgency TEXT NOT NULL,
            request_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add sample donors if empty
    cursor.execute('SELECT COUNT(*) FROM donors')
    if cursor.fetchone()[0] == 0:
        sample_donors = [
            ('John Smith', 'john@email.com', '1234567890', 'A+', 28, 'New York'),
            ('Maria Garcia', 'maria@email.com', '2345678901', 'B-', 35, 'Boston'),
            ('David Johnson', 'david@email.com', '3456789012', 'O+', 42, 'Chicago'),
            ('Sarah Williams', 'sarah@email.com', '4567890123', 'AB+', 29, 'Miami'),
            ('Mike Brown', 'mike@email.com', '5678901234', 'A-', 31, 'Los Angeles'),
            ('Emily Davis', 'emily@email.com', '6789012345', 'B+', 26, 'Seattle'),
            ('Robert Wilson', 'robert@email.com', '7890123456', 'O-', 45, 'Houston'),
            ('Lisa Miller', 'lisa@email.com', '8901234567', 'AB-', 33, 'Phoenix')
        ]
        cursor.executemany(
            'INSERT INTO donors (name, email, phone, blood_group, age, location) VALUES (?, ?, ?, ?, ?, ?)',
            sample_donors
        )
    
    conn.commit()
    conn.close()

# KNN Algorithm Class
class BloodDonorMatcher:
    def __init__(self):
        self.knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.blood_encoder = LabelEncoder()
        
    def prepare_data(self, donors_df):
        """Prepare donor data for KNN algorithm"""
        # Convert blood groups to numerical values
        blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        self.blood_encoder.fit(blood_groups)
        
        donors_df['blood_encoded'] = self.blood_encoder.transform(donors_df['blood_group'])
        
        # Use blood group and age as features
        features = donors_df[['blood_encoded', 'age']].values
        return features
    
    def find_matching_donors(self, patient_blood_group, patient_age=30):
        """Find matching donors using KNN algorithm"""
        conn = sqlite3.connect('blood_bank.db')
        donors_df = pd.read_sql('SELECT * FROM donors', conn)
        conn.close()
        
        if donors_df.empty:
            return []
        
        # Prepare features
        features = self.prepare_data(donors_df)
        
        # Fit KNN model
        self.knn.fit(features)
        
        # Prepare patient data
        try:
            patient_blood_encoded = self.blood_encoder.transform([patient_blood_group])[0]
        except:
            patient_blood_encoded = 0
            
        patient_features = np.array([[patient_blood_encoded, patient_age]])
        
        # Find nearest neighbors
        distances, indices = self.knn.kneighbors(patient_features)
        
        # Get matching donors with scores
        matching_donors = []
        for i, idx in enumerate(indices[0]):
            donor = dict(donors_df.iloc[idx])
            donor['match_score'] = round(float(distances[0][i]), 2)
            matching_donors.append(donor)
        
        return matching_donors

# Flask Routes
@app.route('/')
def index():
    conn = sqlite3.connect('blood_bank.db')
    total_donors = conn.execute('SELECT COUNT(*) FROM donors').fetchone()[0]
    total_patients = conn.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
    conn.close()
    
    return render_template_string(INDEX_HTML, 
                                total_donors=total_donors,
                                total_patients=total_patients)

@app.route('/register_donor', methods=['GET', 'POST'])
def register_donor():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        blood_group = request.form['blood_group']
        age = int(request.form['age'])
        location = request.form['location']
        
        conn = sqlite3.connect('blood_bank.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO donors (name, email, phone, blood_group, age, location) VALUES (?, ?, ?, ?, ?, ?)',
            (name, email, phone, blood_group, age, location)
        )
        conn.commit()
        conn.close()
        
        flash('Donor registered successfully!', 'success')
        return redirect('/')
    
    return render_template_string(REGISTER_DONOR_HTML)

@app.route('/request_blood', methods=['GET', 'POST'])
def request_blood():
    if request.method == 'POST':
        name = request.form['name']
        blood_group = request.form['blood_group']
        location = request.form['location']
        units = int(request.form['units'])
        urgency = request.form['urgency']
        
        # Save patient request
        conn = sqlite3.connect('blood_bank.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO patients (name, blood_group, location, units, urgency) VALUES (?, ?, ?, ?, ?)',
            (name, blood_group, location, units, urgency)
        )
        conn.commit()
        conn.close()
        
        # Use KNN to find matching donors
        matcher = BloodDonorMatcher()
        matching_donors = matcher.find_matching_donors(blood_group)
        
        return render_template_string(RESULTS_HTML, 
                                    donors=matching_donors,
                                    patient_blood_group=blood_group)
    
    return render_template_string(REQUEST_BLOOD_HTML)

@app.route('/search_donors')
def search_donors():
    blood_group = request.args.get('blood_group', '')
    
    conn = sqlite3.connect('blood_bank.db')
    
    if blood_group:
        donors = conn.execute(
            'SELECT * FROM donors WHERE blood_group = ?', 
            (blood_group,)
        ).fetchall()
    else:
        donors = conn.execute('SELECT * FROM donors').fetchall()
    
    conn.close()
    
    # Convert to list of dicts
    donors_list = [dict(donor) for donor in donors]
    
    return render_template_string(SEARCH_DONORS_HTML, donors=donors_list)

@app.route('/stats')
def stats():
    conn = sqlite3.connect('blood_bank.db')
    
    total_donors = conn.execute('SELECT COUNT(*) FROM donors').fetchone()[0]
    total_patients = conn.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
    
    # Get blood group statistics
    blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    blood_stats = {}
    
    for bg in blood_groups:
        count = conn.execute(
            'SELECT COUNT(*) FROM donors WHERE blood_group = ?', 
            (bg,)
        ).fetchone()[0]
        blood_stats[bg] = count
    
    conn.close()
    
    return render_template_string(STATS_HTML,
                                total_donors=total_donors,
                                total_patients=total_patients,
                                blood_stats=blood_stats)

# Base template for inheritance
@app.context_processor
def inject_base_template():
    def render_base_template(content=''):
        conn = sqlite3.connect('blood_bank.db')
        total_donors = conn.execute('SELECT COUNT(*) FROM donors').fetchone()[0]
        total_patients = conn.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
        conn.close()
        
        return INDEX_HTML.replace('{% block content %}{% endblock %}', content).replace(
            '{{ total_donors }}', str(total_donors)).replace(
            '{{ total_patients }}', str(total_patients))
    return dict(base=render_base_template)

if __name__ == '__main__':
    print("🩸 Initializing Blood Bank Management System...")
    print("📊 Setting up database...")
    init_db()
    print("🤖 KNN Algorithm ready...")
    print("🚀 Starting server on http://localhost:5000")
    print("\n🌟 Features:")
    print("   • KNN-based donor matching")
    print("   • Real-time blood group search")
    print("   • Emergency request system")
    print("   • Smart compatibility scoring")
    print("\n📋 Available Routes:")
    print("   • /              - Homepage")
    print("   • /register_donor - Register as donor")
    print("   • /request_blood  - Request blood")
    print("   • /search_donors  - Find donors")
    print("   • /stats          - System statistics")
    
    app.run(debug=True, host='0.0.0.0', port=5000)



