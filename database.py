# database.py
import sqlite3
import hashlib
import os
from datetime import datetime
from collections import Counter # Import needed for stability calculation

DATABASE = 'sleep_app.db'

def init_db():
    """Initialize the database with the schema."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            gender TEXT NOT NULL,
            age INTEGER NOT NULL,
            occupation TEXT NOT NULL,
            sleep_duration REAL NOT NULL,
            quality_of_sleep INTEGER NOT NULL,
            physical_activity_level INTEGER NOT NULL,
            stress_level INTEGER NOT NULL,
            heart_rate INTEGER NOT NULL,
            daily_steps INTEGER NOT NULL,
            bmi_category TEXT NOT NULL,
            blood_pressure TEXT NOT NULL,
            prediction_result TEXT NOT NULL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            entry_date DATE NOT NULL,
            sleep_duration REAL,
            quality_of_sleep INTEGER,
            stress_level INTEGER,
            physical_activity_level INTEGER,
            heart_rate INTEGER,
            daily_steps INTEGER,
            mood TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, entry_date)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_tracking_user_id ON daily_tracking(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_tracking_date ON daily_tracking(entry_date)")
    conn.commit()
    conn.close()

def hash_password(password):
    salt = os.urandom(32)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt + pwdhash

def verify_password(stored_password, provided_password):
    salt = stored_password[:32]
    stored_pwdhash = stored_password[32:]
    pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
    return pwdhash == stored_pwdhash

def register_user(email, password, first_name=None, last_name=None):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        hashed_pw = hash_password(password)
        cursor.execute("""
            INSERT INTO users (email, password_hash, first_name, last_name)
            VALUES (?, ?, ?, ?)
        """, (email, hashed_pw, first_name, last_name))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def login_user(email, password):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user_row = cursor.fetchone()
    conn.close()
    if user_row and verify_password(user_row['password_hash'], password):
        return {
            'id': user_row['id'],
            'email': user_row['email'],
            'first_name': user_row['first_name'],
            'last_name': user_row['last_name'],
            'created_at': user_row['created_at']
        }
    return None

def get_user_by_id(user_id):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_row = cursor.fetchone()
    conn.close()
    if user_row:
        return {
            'id': user_row['id'],
            'email': user_row['email'],
            'first_name': user_row['first_name'],
            'last_name': user_row['last_name'],
            'created_at': user_row['created_at']
        }
    return None

def update_user_profile(user_id, first_name=None, last_name=None):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        updates = []
        params = []
        if first_name is not None:
            updates.append("first_name = ?")
            params.append(first_name)
        if last_name is not None:
            updates.append("last_name = ?")
            params.append(last_name)
        if not updates:
            return True
        params.append(user_id)
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating profile: {e}")
        return False

def save_prediction(user_id, form_data, prediction_result):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions
            (user_id, gender, age, occupation, sleep_duration, quality_of_sleep,
             physical_activity_level, stress_level, heart_rate, daily_steps,
             bmi_category, blood_pressure, prediction_result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, form_data['Gender'], form_data['Age'], form_data['Occupation'],
            form_data['Sleep Duration'], form_data['Quality of Sleep'],
            form_data['Physical Activity'], form_data['Stress Level'],
            form_data['Heart Rate'], form_data['Daily Steps'],
            form_data['BMI Category'], form_data['Blood Pressure'],
            prediction_result
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

def get_user_predictions(user_id, limit=100, offset=0):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE user_id = ?
        ORDER BY prediction_date DESC
        LIMIT ? OFFSET ?
    """, (user_id, limit, offset))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_user_prediction_count(user_id):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_user_prediction_by_id(user_id, prediction_id):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE user_id = ? AND id = ?
    """, (user_id, prediction_id))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def delete_user_prediction(user_id, prediction_id):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM predictions
            WHERE id = ? AND user_id = ?
        """, (prediction_id, user_id))
        conn.commit()
        deleted_rows = cursor.rowcount
        conn.close()
        return deleted_rows > 0
    except Exception as e:
        print(f"Error deleting prediction: {e}")
        return False

def get_dashboard_stats(user_id):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    stats = {}

    # --- Summary Stats ---
    cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE user_id = ?", (user_id,))
    stats['total_predictions'] = cursor.fetchone()['count'] or 0
    cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE user_id = ? AND prediction_result = 'Healthy'", (user_id,))
    stats['healthy_count'] = cursor.fetchone()['count'] or 0
    cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE user_id = ? AND prediction_result = 'Insomnia'", (user_id,))
    stats['insomnia_count'] = cursor.fetchone()['count'] or 0
    cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE user_id = ? AND prediction_result = 'Sleep Apnea'", (user_id,))
    stats['apnea_count'] = cursor.fetchone()['count'] or 0
    stats['prediction_distribution'] = {
        "labels": ["Healthy", "Insomnia", "Sleep Apnea"],
        "data": [stats['healthy_count'], stats['insomnia_count'], stats['apnea_count']]
    }

    # --- Sleep Duration Trend (Last 7) ---
    cursor.execute("""
        SELECT prediction_date, sleep_duration 
        FROM predictions 
        WHERE user_id = ? 
        ORDER BY prediction_date DESC 
        LIMIT 7
    """, (user_id,))
    duration_rows = cursor.fetchall()
    # Reverse to show oldest first on chart
    duration_rows = list(duration_rows)
    duration_rows.reverse()
    stats['duration_trend'] = {
        "labels": [row['prediction_date'] for row in duration_rows],
        "data": [row['sleep_duration'] for row in duration_rows]
    }

    # --- Stress vs Quality (Last 10) ---
    cursor.execute("""
        SELECT stress_level, quality_of_sleep 
        FROM predictions 
        WHERE user_id = ? 
        ORDER BY prediction_date DESC 
        LIMIT 10
    """, (user_id,))
    stress_quality_rows = cursor.fetchall()
    # Reverse to show oldest first on chart
    stress_quality_rows = list(stress_quality_rows)
    stress_quality_rows.reverse()
    stats['stress_quality_data'] = {
        "stress": [row['stress_level'] for row in stress_quality_rows],
        "quality": [row['quality_of_sleep'] for row in stress_quality_rows]
    }

    # --- Recent Activity (Last 5) ---
    cursor.execute("""
        SELECT prediction_result, prediction_date 
        FROM predictions 
        WHERE user_id = ? 
        ORDER BY prediction_date DESC 
        LIMIT 5
    """, (user_id,))
    activity_rows = cursor.fetchall()
    stats['recent_activity'] = [
        {
            "action": "Prediction",
            "details": row['prediction_result'],
            "date": row['prediction_date']
        } for row in activity_rows
    ]

    # --- Latest Metrics for Cards ---
    cursor.execute("""
        SELECT stress_level, heart_rate, blood_pressure, daily_steps
        FROM predictions
        WHERE user_id = ?
        ORDER BY prediction_date DESC
        LIMIT 1
    """, (user_id,))
    latest_metrics = cursor.fetchone()
    if latest_metrics:
        stats['latest_stress'] = latest_metrics['stress_level']
        stats['latest_heart_rate'] = latest_metrics['heart_rate']
        stats['latest_blood_pressure'] = latest_metrics['blood_pressure']
        stats['latest_daily_steps'] = latest_metrics['daily_steps']
    else:
        stats['latest_stress'] = 'N/A'
        stats['latest_heart_rate'] = 'N/A'
        stats['latest_blood_pressure'] = 'N/A'
        stats['latest_daily_steps'] = 'N/A'

    # --- Historical Data for New Charts ---
    cursor.execute("""
        SELECT prediction_date, stress_level, heart_rate, 
               SUBSTR(blood_pressure, 1, INSTR(blood_pressure, '/') - 1) as systolic,
               SUBSTR(blood_pressure, INSTR(blood_pressure, '/') + 1) as diastolic,
               daily_steps
        FROM predictions
        WHERE user_id = ?
        ORDER BY prediction_date DESC
        LIMIT 10
    """, (user_id,))
    historical_data = cursor.fetchall()
    if historical_data:
        # Reverse to show oldest first on chart
        historical_data = list(historical_data)
        historical_data.reverse()
        stats['historical_data'] = {
            "dates": [row['prediction_date'] for row in historical_data],
            "stress": [row['stress_level'] for row in historical_data],
            "heart_rate": [row['heart_rate'] for row in historical_data],
            "systolic_bp": [int(row['systolic']) if row['systolic'] and row['systolic'].isdigit() else 0 for row in historical_data],
            "diastolic_bp": [int(row['diastolic']) if row['diastolic'] and row['diastolic'].isdigit() else 0 for row in historical_data],
            "daily_steps": [row['daily_steps'] for row in historical_data]
        }
    else:
        stats['historical_data'] = {
            "dates": [],
            "stress": [],
            "heart_rate": [],
            "systolic_bp": [],
            "diastolic_bp": [],
            "daily_steps": []
        }

    # --- Prediction Stability (New Metric) ---
    # Calculate stability based on the last N predictions (e.g., 10)
    stability_window = 10
    cursor.execute("""
        SELECT prediction_result
        FROM predictions
        WHERE user_id = ?
        ORDER BY prediction_date DESC
        LIMIT ?
    """, (user_id, stability_window))
    recent_preds_for_stability = [row['prediction_result'] for row in cursor.fetchall()]

    if recent_preds_for_stability:
        # Calculate stability: percentage of predictions matching the mode
        counts = Counter(recent_preds_for_stability)
        most_common_count = counts.most_common(1)[0][1] if counts else 0
        stability_percentage = (most_common_count / len(recent_preds_for_stability)) * 100
        stats['prediction_stability'] = {
            "labels": [f"Last {len(recent_preds_for_stability)} Predictions"],
            "data": [round(stability_percentage, 1)] # Round to 1 decimal place
        }
    else:
        stats['prediction_stability'] = {
            "labels": [],
            "data": []
        }

    conn.close()
    return stats

def save_daily_tracking_from_csv(user_id, df, filename):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        saved_count = 0
        for index, row in df.iterrows():
            # Heuristic: Use current date or a date column if it exists
            entry_date = datetime.now().date()
            sleep_duration = row.get('Sleep Duration', None)
            quality_of_sleep = row.get('Quality of Sleep', None)
            stress_level = row.get('Stress Level', None)
            heart_rate = row.get('Heart Rate', None)
            daily_steps = row.get('Daily Steps', None)
            physical_activity_level = row.get('Physical Activity Level', None) # If exists in CSV
            notes = f"Uploaded from {filename}"
            cursor.execute("""
                INSERT OR IGNORE INTO daily_tracking
                (user_id, entry_date, sleep_duration, quality_of_sleep, stress_level,
                 physical_activity_level, heart_rate, daily_steps, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, entry_date, sleep_duration, quality_of_sleep, stress_level,
                  physical_activity_level, heart_rate, daily_steps, notes))
            if cursor.rowcount > 0:
                saved_count += 1
        conn.commit()
        conn.close()
        return saved_count
    except Exception as e:
        print(f"Error saving daily tracking: {e}")
        return 0

# Initialize the database when the module is imported
init_db()
