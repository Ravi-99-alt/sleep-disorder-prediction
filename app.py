from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import joblib
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
from database import *
from functools import wraps

app = Flask(__name__)
app.secret_key = "sleep_app_secret_key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- MODEL LOADING ---------------- #

try:
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    gender_encoder = joblib.load("models/gender_encoder.pkl")
    occupation_encoder = joblib.load("models/occupation_encoder.pkl")
    bmi_category_encoder = joblib.load("models/bmi_category_encoder.pkl")
    target_encoder = joblib.load("models/target_encoder.pkl")

    gender_classes = list(gender_encoder.classes_)
    occupation_classes = list(occupation_encoder.classes_)
    bmi_classes = list(bmi_category_encoder.classes_)

    blood_pressure_classes = [
        "120/80","130/85","140/90","110/70",
        "115/75","125/80","135/85"
    ]

except Exception as e:
    raise SystemExit(f"Error loading model files: {e}")


# ---------------- LOGIN REQUIRED ---------------- #

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "logged_in" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap


# ---------------- HOME ---------------- #

@app.route("/")
def home():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return render_template("home.html")


# ---------------- LOGIN ---------------- #

@app.route("/login", methods=["GET","POST"])
def login():
    error=None
    if request.method=="POST":

        email=request.form.get("email")
        password=request.form.get("password")

        user=login_user(email,password)

        if user:
            session["logged_in"]=True
            session["user_id"]=user["id"]
            return redirect(url_for("dashboard"))
        else:
            error="Invalid email or password"

    return render_template("login.html",error=error)


# ---------------- REGISTER ---------------- #

@app.route("/register",methods=["GET","POST"])
def register():

    error=None
    success=None

    if request.method=="POST":

        email=request.form.get("email")
        password=request.form.get("password")
        confirm=request.form.get("confirm_password")

        if not email or not password:
            error="Email and password required"

        elif password!=confirm:
            error="Passwords do not match"

        else:
            if register_user(email,password):
                success="Registration successful"
            else:
                error="Email already exists"

    return render_template("register.html",error=error,success=success)


# ---------------- LOGOUT ---------------- #

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ---------------- DASHBOARD ---------------- #

@app.route("/dashboard")
@login_required
def dashboard():

    user_id=session["user_id"]

    user=get_user_by_id(user_id)

    stats=get_dashboard_stats(user_id)

    recent_predictions=get_user_predictions(user_id,limit=5)

    return render_template(
        "dashboard.html",
        user=user,
        stats=stats,
        recent_predictions=recent_predictions
    )


# ---------------- UPLOAD CSV ---------------- #

@app.route("/upload",methods=["GET","POST"])
@login_required
def upload():

    if request.method=="POST":

        file=request.files.get("document")

        if not file or file.filename=="":
            flash("No file selected","warning")
            return redirect(url_for("upload"))

        filename=secure_filename(file.filename)

        if not filename.endswith(".csv"):
            flash("Upload CSV file only","danger")
            return redirect(url_for("upload"))

        file_path=os.path.join(app.config["UPLOAD_FOLDER"],filename)
        file.save(file_path)

        return redirect(url_for("preview",filename=filename))

    return render_template("upload.html")


# ---------------- CSV PREVIEW ---------------- #

@app.route("/preview/<filename>")
@login_required
def preview(filename):

    file_path=os.path.join(app.config["UPLOAD_FOLDER"],filename)

    if not os.path.exists(file_path):
        flash("File not found","danger")
        return redirect(url_for("upload"))

    df=pd.read_csv(file_path,nrows=5)

    headers=df.columns.tolist()

    return render_template("preview.html",filename=filename,headers=headers)


# ---------------- PREDICTION PAGE ---------------- #

@app.route("/index")
@login_required
def index():

    return render_template(
        "index.html",
        gender_classes=gender_classes,
        occupation_classes=occupation_classes,
        bmi_classes=bmi_classes,
        blood_pressure_classes=blood_pressure_classes
    )


# ---------------- ML PREDICTION ---------------- #

@app.route("/predict",methods=["POST"])
@login_required
def predict():

    try:

        gender=request.form["Gender"]
        age=int(request.form["Age"])
        occupation=request.form["Occupation"]

        sleep_duration=float(request.form["Sleep_duration"])
        quality=int(request.form["Quality_of_sleep"])

        physical=float(request.form["Physical_activity"])
        stress=int(request.form["Stress_Level"])

        heart_rate=int(request.form["Heart_rate"])
        steps=int(request.form["Daily_steps"])

        bmi=request.form["BMI_category"]

        bp=request.form["Blood_pressure"]

        systolic,diastolic=map(int,bp.split("/"))

        gender_enc=gender_encoder.transform([gender])[0]
        occ_enc=occupation_encoder.transform([occupation])[0]
        bmi_enc=bmi_category_encoder.transform([bmi])[0]

        input_data=np.array([[
            gender_enc,
            age,
            occ_enc,
            sleep_duration,
            quality,
            physical,
            stress,
            heart_rate,
            steps,
            bmi_enc,
            systolic,
            diastolic
        ]])

        input_scaled=scaler.transform(input_data)

        prediction=model.predict(input_scaled)[0]

        predicted_label=target_encoder.inverse_transform([prediction])[0]

        form_data={
            "Gender":gender,
            "Age":age,
            "Occupation":occupation,
            "Sleep Duration":sleep_duration,
            "Quality of Sleep":quality,
            "Physical Activity":physical,
            "Stress Level":stress,
            "Heart Rate":heart_rate,
            "Daily Steps":steps,
            "BMI Category":bmi,
            "Blood Pressure":bp
        }

        save_prediction(session["user_id"],form_data,predicted_label)

        return render_template(
            "result.html",
            prediction=predicted_label,
            form_data=form_data
        )

    except Exception as e:

        flash(f"Prediction error: {e}","danger")

        return redirect(url_for("index"))


# ---------------- HISTORY ---------------- #

@app.route("/history")
@login_required
def history():
    return render_template("history.html")


# ---------------- HISTORY DATA ---------------- #

@app.route("/history_data",methods=["POST"])
@login_required
def history_data():

    user_id=session["user_id"]

    predictions=get_user_predictions(user_id)

    return jsonify(predictions)


# ---------------- PROFILE ---------------- #

@app.route("/profile",methods=["GET","POST"])
@login_required
def profile():

    user_id=session["user_id"]

    user=get_user_by_id(user_id)

    error=None
    success=None

    if request.method=="POST":

        first=request.form.get("first_name")
        last=request.form.get("last_name")

        if update_user_profile(user_id,first,last):
            success="Profile updated"
        else:
            error="Update failed"

    return render_template(
        "profile.html",
        user=user,
        error=error,
        success=success
    )


# ---------------- MAIN ---------------- #

if __name__=="__main__":
    app.run(debug=True)
