# app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import joblib
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
from database import *

app = Flask(__name__)
app.secret_key = "a_very_secret_key_for_sleep_app_2024"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Model and Preprocessing Tools ---
try:
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    gender_encoder = joblib.load('models/gender_encoder.pkl')
    occupation_encoder = joblib.load('models/occupation_encoder.pkl')
    bmi_category_encoder = joblib.load('models/bmi_category_encoder.pkl')
    target_encoder = joblib.load('models/target_encoder.pkl')
    gender_classes = list(gender_encoder.classes_)
    occupation_classes = list(occupation_encoder.classes_)
    bmi_classes = list(bmi_category_encoder.classes_)
    blood_pressure_classes = ["120/80", "130/85", "140/90", "110/70", "115/75", "125/80", "135/85"]
except Exception as e:
    raise SystemExit(f"Error loading model or encoders: {e}")


def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect(url_for('login'))
    return wrap


@app.route('/')
def home():
    if 'logged_in' in session:
        return redirect(url_for('dashboard'))
    return render_template("home.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = login_user(email, password)
        if user:
            session['logged_in'] = True
            session['user_id'] = user['id']
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid email or password."
    return render_template("login.html", error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if not email or not password:
            error = "Email and password are required."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif len(password) < 6:
            error = "Password must be at least 6 characters long."
        else:
            if register_user(email, password):
                success = "Registration successful! You can now log in."
            else:
                error = "Email address already exists."
    return render_template("register.html", error=error, success=success)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    user = get_user_by_id(user_id)
    if not user:
        session.clear()
        return redirect(url_for('login'))
    stats = get_dashboard_stats(user_id)
    recent_predictions = get_user_predictions(user_id, limit=5)
    return render_template("dashboard.html", user=user, stats=stats, recent_predictions=recent_predictions)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files.get('document')
        if not file or file.filename == '':
            flash("No file selected.", "warning")
            return redirect(url_for('upload'))
        filename = secure_filename(file.filename)
        if not filename.endswith('.csv'):
            flash("Please upload a CSV file.", "danger")
            return redirect(url_for('upload'))
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('preview', filename=filename))
        except Exception as e:
            flash(f"Upload error: {str(e)}", "danger")
            return redirect(url_for('upload'))
    return render_template("upload.html")


@app.route('/preview/<filename>')
@login_required
def preview(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        flash("File not found.", "warning")
        return redirect(url_for('upload'))
    try:
        df = pd.read_csv(file_path, nrows=1)
        headers = df.columns.tolist()
    except Exception as e:
        flash(f"Error reading file: {str(e)}", "danger")
        headers = []
    return render_template("preview.html", filename=filename, headers=headers)


@app.route('/data_server/<filename>', methods=['POST'])
@login_required
def data_server(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        df = pd.read_csv(file_path)
        draw = int(request.form.get('draw', 1))
        start = int(request.form.get('start', 0))
        length = int(request.form.get('length', 10))
        search_value = request.form.get('search[value]', '').lower()
        if search_value:
            mask = df.apply(lambda row: row.astype(str).str.lower().str.contains(search_value).any(), axis=1)
            df_filtered = df[mask]
        else:
            df_filtered = df

        records_total = len(df_filtered)
        df_page = df_filtered.iloc[start:start + length]

        return jsonify({
            'draw': draw,
            'recordsTotal': len(df),
            'recordsFiltered': records_total,
            'data': df_page.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/save_daily_tracking/<filename>', methods=['POST'])
@login_required
def save_daily_tracking(filename):
    user_id = session.get('user_id')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        flash("File not found.", "warning")
        return redirect(url_for('upload'))
    try:
        df = pd.read_csv(file_path)
        saved_count = save_daily_tracking_from_csv(user_id, df, filename)
        flash(f"Successfully saved {saved_count} entries to your daily tracking.", "success")
    except Exception as e:
        flash(f"Error saving entries: {str(e)}", "danger")
    return redirect(url_for('dashboard'))


@app.route('/prefill_index', methods=['POST'])
@login_required
def prefill_index():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        session['prefilled_data'] = {
            'Gender': data.get('Gender'),
            'Age': data.get('Age'),
            'Occupation': data.get('Occupation'),
            'Sleep Duration': data.get('Sleep Duration'),
            'Quality of Sleep': data.get('Quality of Sleep'),
            'Physical Activity': data.get('Physical Activity'),
            'Stress Level': data.get('Stress Level'),
            'Heart Rate': data.get('Heart Rate'),
            'Daily Steps': data.get('Daily Steps'),
            'BMI Category': data.get('BMI Category'),
            'Blood Pressure': data.get('Blood Pressure')
        }
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/index', methods=['GET'])
@login_required
def index():
    prefilled_data = session.pop('prefilled_data', None)
    return render_template(
        "index.html",
        gender_classes=gender_classes,
        occupation_classes=occupation_classes,
        bmi_classes=bmi_classes,
        prefilled_data=prefilled_data
    )


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        form_data = {
            "Gender": request.form['Gender'],
            "Age": int(request.form['Age']),
            "Occupation": request.form['Occupation'],
            "Sleep Duration": float(request.form['Sleep_duration']),
            "Quality of Sleep": int(request.form['Quality_of_sleep']),
            "Physical Activity": float(request.form['Physical_activity']),
            "Stress Level": int(request.form['Stress_Level']),
            "Heart Rate": int(request.form['Heart_rate']),
            "Daily Steps": int(request.form['Daily_steps']),
            "BMI Category": request.form['BMI_category'],
            "Blood Pressure": request.form['Blood_pressure']
        }
        try:
            systolic, diastolic = map(int, form_data["Blood Pressure"].strip().split('/'))
        except Exception:
            flash("Invalid Blood Pressure Format. Use Systolic/Diastolic (e.g., 120/80).", "danger")
            return redirect(url_for('index'))

        gender_enc = gender_encoder.transform([form_data["Gender"]])[0]
        occupation_enc = occupation_encoder.transform([form_data["Occupation"]])[0]
        bmi_enc = bmi_category_encoder.transform([form_data["BMI Category"]])[0]

        input_data = np.array([[gender_enc, form_data["Age"], occupation_enc, form_data["Sleep Duration"],
                                form_data["Quality of Sleep"], form_data["Physical Activity"], form_data["Stress Level"],
                                form_data["Heart Rate"], form_data["Daily Steps"], bmi_enc, systolic, diastolic]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        predicted_label = target_encoder.inverse_transform([prediction])[0]

        user_id = session['user_id']
        save_prediction(user_id, form_data, predicted_label)

        suggestions = {
            "Healthy": {
                "title": "Healthy!",
                "message": "Your sleep metrics are within healthy ranges. Keep up the great work!",
                "tips": [
                    "Maintain your current sleep schedule for consistency.",
                    "Continue with your physical activity routine.",
                    "Practice mindfulness or meditation before bed to enhance relaxation.",
                    "Keep a sleep diary to track long-term trends."
                ]
            },
            "Insomnia": {
                "title": "Insomnia",
                "message": "Your results suggest signs of insomnia. These steps can help improve your sleep quality.",
                "tips": [
                    "Establish a consistent bedtime and wake-up time, even on weekends.",
                    "Create a relaxing bedtime routine (e.g., reading, taking a warm bath).",
                    "Avoid screens (phones, tablets, computers) for at least an hour before bed.",
                    "Keep your bedroom cool, dark, and quiet.",
                    "Limit caffeine and alcohol, especially in the evening.",
                    "Consider cognitive behavioral therapy for insomnia (CBT-I)."
                ]
            },
            "Sleep Apnea": {
                "title": "Potential Sleep Apnea Risk",
                "message": "Your results indicate a potential risk for sleep apnea. It's important to seek professional medical advice.",
                "tips": [
                    "Consult a healthcare professional or a sleep specialist for a proper diagnosis.",
                    "If overweight, even a small amount of weight loss can significantly improve symptoms.",
                    "Try sleeping on your side instead of your back.",
                    "Avoid alcohol and sedatives, as they can relax throat muscles and worsen apnea.",
                    "Keep nasal passages open using a saline spray or a humidifier.",
                    "Continuous Positive Airway Pressure (CPAP) therapy is a common and effective treatment."
                ]
            }
        }
        suggestion_data = suggestions.get(predicted_label, {
            "title": "General Advice",
            "message": "Please consult a healthcare professional for personalized advice based on your results.",
            "tips": ["Maintain a healthy lifestyle.", "Monitor your sleep patterns.", "Seek professional help if symptoms persist."]
        })

        return render_template("result.html", prediction=predicted_label, form_data=form_data, suggestion=suggestion_data)
    except Exception as e:
        flash(f"Error in prediction: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/history')
@login_required
def history():
    return render_template("history.html")


@app.route('/history_data', methods=['POST'])
@login_required
def history_data():
    user_id = session['user_id']
    draw = int(request.form.get('draw', 1))
    start = int(request.form.get('start', 0))
    length = int(request.form.get('length', 10))

    predictions_page = get_user_predictions(user_id, limit=length, offset=start)
    records_total = get_user_prediction_count(user_id)

    data = []
    for row in predictions_page:
        data.append({
            "id": row["id"],
            "prediction_date": row["prediction_date"],
            "age": row["age"],
            "gender": row["gender"],
            "occupation": row["occupation"],
            "sleep_duration": row["sleep_duration"],
            "quality_of_sleep": row["quality_of_sleep"],
            "stress_level": row["stress_level"],
            "bmi_category": row["bmi_category"],
            "prediction_result": row["prediction_result"]
        })

    return jsonify({
        'draw': draw,
        'recordsTotal': records_total,
        'recordsFiltered': records_total,
        'data': data
    })


@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    user_id = session['user_id']
    success = delete_user_prediction(user_id, prediction_id)
    if success:
        return jsonify({"success": True, "message": "Prediction deleted successfully."})
    else:
        return jsonify({"success": False, "message": "Failed to delete prediction or prediction not found."}), 400


@app.route('/view_prediction/<int:prediction_id>')
@login_required
def view_prediction(prediction_id):
    user_id = session['user_id']
    prediction_data = get_user_prediction_by_id(user_id, prediction_id)
    if not prediction_data:
        flash("Prediction not found or access denied.", "danger")
        return redirect(url_for('dashboard'))

    form_data = {
        "Gender": prediction_data["gender"],
        "Age": prediction_data["age"],
        "Occupation": prediction_data["occupation"],
        "Sleep Duration": prediction_data["sleep_duration"],
        "Quality of Sleep": prediction_data["quality_of_sleep"],
        "Physical Activity": prediction_data["physical_activity_level"],
        "Stress Level": prediction_data["stress_level"],
        "Heart Rate": prediction_data["heart_rate"],
        "Daily Steps": prediction_data["daily_steps"],
        "BMI Category": prediction_data["bmi_category"],
        "Blood Pressure": prediction_data["blood_pressure"]
    }
    predicted_label = prediction_data["prediction_result"]

    suggestions = {
        "Healthy": {
            "title": "Excellent Work!",
            "message": "Your sleep metrics are within healthy ranges. Keep up the great work!",
            "tips": [
                "Maintain your current sleep schedule for consistency.",
                "Continue with your physical activity routine.",
                "Practice mindfulness or meditation before bed to enhance relaxation.",
                "Keep a sleep diary to track long-term trends."
            ]
        },
        "Insomnia": {
            "title": "Managing Insomnia",
            "message": "Your results suggest signs of insomnia. These steps can help improve your sleep quality.",
            "tips": [
                "Establish a consistent bedtime and wake-up time, even on weekends.",
                "Create a relaxing bedtime routine (e.g., reading, taking a warm bath).",
                "Avoid screens (phones, tablets, computers) for at least an hour before bed.",
                "Keep your bedroom cool, dark, and quiet.",
                "Limit caffeine and alcohol, especially in the evening.",
                "Consider cognitive behavioral therapy for insomnia (CBT-I)."
            ]
        },
        "Sleep Apnea": {
            "title": "Potential Sleep Apnea Risk",
            "message": "Your results indicate a potential risk for sleep apnea. It's important to seek professional medical advice.",
            "tips": [
                "Consult a healthcare professional or a sleep specialist for a proper diagnosis.",
                "If overweight, even a small amount of weight loss can significantly improve symptoms.",
                "Try sleeping on your side instead of your back.",
                "Avoid alcohol and sedatives, as they can relax throat muscles and worsen apnea.",
                "Keep nasal passages open using a saline spray or a humidifier.",
                "Continuous Positive Airway Pressure (CPAP) therapy is a common and effective treatment."
            ]
        }
    }
    suggestion_data = suggestions.get(predicted_label, {
        "title": "General Advice",
        "message": "Please consult a healthcare professional for personalized advice based on your results.",
        "tips": ["Maintain a healthy lifestyle.", "Monitor your sleep patterns.", "Seek professional help if symptoms persist."]
    })

    return render_template("result.html", prediction=predicted_label, form_data=form_data, suggestion=suggestion_data, is_history_view=True)


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_id = session['user_id']
    user = get_user_by_id(user_id)
    if not user:
        session.clear()
        return redirect(url_for('login'))
    error = None
    success = None
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        if update_user_profile(user_id, first_name, last_name):
            success = "Profile updated successfully!"
            user = get_user_by_id(user_id)
        else:
            error = "Failed to update profile."
    return render_template("profile.html", user=user, error=error, success=success)


if __name__ == '__main__':
    app.run(debug=True)
