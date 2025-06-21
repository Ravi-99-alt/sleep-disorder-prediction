from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and preprocessing tools
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

    # Add Blood Pressure Options
    blood_pressure_classes = ["120/80", "130/85", "140/90", "110/70", "115/75", "125/80", "135/85"]

except Exception as e:
    raise SystemExit(f"Error loading model or encoders: {e}")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == "admin" and password == "1234":
            session['logged_in'] = True
            return redirect(url_for('user'))
        else:
            error = "❌ Invalid username or password"
    return render_template("login.html", error=error)

@app.route('/user')
def user():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template("user.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('document')
        if not file or file.filename == '':
            return "<h3 class='text-danger'>❌ No file selected</h3>"

        try:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('preview', filename=filename))
        except Exception as e:
            return f"<h3 class='text-danger'>⚠️ Upload error: {str(e)}</h3><br><a href='/upload' class='btn btn-outline-light'>Go Back</a>"

    return render_template("upload.html")

@app.route('/preview/<filename>')
def preview(filename):
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        return "<h3 class='text-warning'>📄 File not found. Please re-upload.</h3>"

    try:
        df = pd.read_csv(file_path, nrows=1)
        headers = df.columns.tolist()
    except Exception:
        headers = []

    return render_template("preview.html", filename=filename, headers=headers)

@app.route('/index', methods=['GET'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    return render_template("index.html",
        gender_classes=gender_classes,
        occupation_classes=occupation_classes,
        bmi_classes=bmi_classes,
        blood_pressure_classes=blood_pressure_classes  # 👈 Added here
    )

@app.route('/data_server/<filename>', methods=['POST'])
def data_server(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        df = pd.read_csv(file_path)
        draw = int(request.form.get('draw', 1))
        start = int(request.form.get('start', 0))
        length = int(request.form.get('length', 10))
        search_value = request.form.get('search[value]', '').lower()

        if search_value:
            df = df[df.apply(lambda row: row.astype(str).str.lower().str.contains(search_value).any(), axis=1)]

        records_total = len(df)
        df_page = df.iloc[start:start + length]

        return jsonify({
            'draw': draw,
            'recordsTotal': records_total,
            'recordsFiltered': records_total,
            'data': df_page.values.tolist()
        })
    except Exception as e:
        return jsonify({
            'draw': 1,
            'recordsTotal': 0,
            'recordsFiltered': 0,
            'data': [],
            'error': str(e)
        })
@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['Gender']
        age = int(request.form['Age'])
        occupation = request.form['Occupation']
        sleep_duration = float(request.form['Sleep_duration'])
        quality_of_sleep = int(request.form['Quality_of_sleep'])
        physical_activity = float(request.form['Physical_activity'])
        stress_level = int(request.form['Stress_Level'])
        heart_rate = int(request.form['Heart_rate'])
        daily_steps = int(request.form['Daily_steps'])
        bmi_category = request.form['BMI_category']
        bp_raw = request.form['Blood_pressure']

        # Parse blood pressure
        try:
            systolic, diastolic = map(int, bp_raw.strip().split('/'))
        except Exception as e:
            return "<h3 class='text-danger'>⚠️ Invalid Blood Pressure Format. Use format like 120/80.</h3>"

        # Encode categorical features
        gender_enc = gender_encoder.transform([gender])[0]
        occupation_enc = occupation_encoder.transform([occupation])[0]
        bmi_enc = bmi_category_encoder.transform([bmi_category])[0]

        # Prepare input data
        input_data = np.array([[gender_enc, age, occupation_enc, sleep_duration,
                                quality_of_sleep, physical_activity, stress_level,
                                heart_rate, daily_steps, bmi_enc, systolic, diastolic]])

        # Scale only if needed
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]  # Use scaled input

        # Pass all form values to result page
        form_data = {
            "Gender": gender,
            "Age": age,
            "Occupation": occupation,
            "Sleep Duration": sleep_duration,
            "Quality of Sleep": quality_of_sleep,
            "Physical Activity": physical_activity,
            "Stress Level": stress_level,
            "Heart Rate": heart_rate,
            "Daily Steps": daily_steps,
            "BMI Category": bmi_category,
            "Blood Pressure": bp_raw
        }

        predicted_label = target_encoder.inverse_transform([prediction])[0]

        return render_template("result.html", prediction=predicted_label, form_data=form_data)

    except Exception as e:
        return f"<h3 class='text-danger'>❌ Error in prediction: {str(e)}</h3>"

if __name__ == '__main__':
    app.run(debug=True)