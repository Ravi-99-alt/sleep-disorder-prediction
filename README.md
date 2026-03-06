# Sleep Disorder Prediction Using Machine Learning

## Project Overview

The Sleep Disorder Prediction System is a Machine Learning–based web application designed to predict potential sleep disorders such as "Insomnia" and "Sleep Apnea" based on a user's health and lifestyle data.

The system analyzes multiple factors including **sleep duration, stress level, heart rate, BMI category, daily steps, and blood pressure** to provide a prediction and helpful health suggestions.

This application also provides an interactive dashboard with visual analytics, allowing users to monitor their sleep health trends.

--------------------------

##  Features

* User Authentication

  * User Registration
  * Login and Logout
  * Profile management

* Machine Learning Prediction

  * Predicts sleep disorders using trained ML model
  * Supports three prediction categories:

    * Healthy
    * Insomnia
    * Sleep Apnea

* Interactive Health Dashboard

  * Sleep Duration Trend
  * Heart Rate & Stress Level Analysis
  * Blood Pressure Monitoring
  * Daily Steps Tracking
  * Prediction Distribution Charts

* CSV Data Upload

  * Upload daily sleep health data
  * Automatically store and analyze health metrics

* Prediction History

  * View previous predictions
  * Delete or review prediction results

------------------------------------

## Machine Learning Model

The model predicts sleep disorders using the following input features:

* Gender
* Age
* Occupation
* Sleep Duration
* Quality of Sleep
* Physical Activity Level
* Stress Level
* Heart Rate
* Daily Steps
* BMI Category
* Blood Pressure

### Algorithms & Tools Used

* Scikit-Learn
* Pandas
* NumPy
* Joblib (for model persistence)

---

## Technologies Used

Backend

* Python
* Flask
* SQLite

Machine Learning

* Scikit-learn
* Pandas
* NumPy

Frontend

* HTML
* CSS
* JavaScript
* Bootstrap
* Chart.js

Other Tools

* Git & GitHub
* VS Code

---

## Project Structure

```
sleep-disorder-prediction
│
├── app.py
|
├── database.py
├── requirements.txt
├── README.md
│
├── models
│   ├── model.pkl
│   ├── scaler.pkl
│
├── templates
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── result.html
│
├── static
│   ├── css
│   ├── js
│
└── uploads
```

---

## Installation & Setup

### 1️ Clone the Repository

```bash
git clone https://github.com/Ravi-99-alt/sleep-disorder-prediction.git
cd sleep-disorder-prediction
```

### 2️ Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

### 3️ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️ Run the Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## Example Prediction Output

The system predicts one of the following:

| Result      | Meaning                            |
| ----------- | ---------------------------------- |
| Healthy     | Normal sleep patterns              |
| Insomnia    | Difficulty sleeping                |
| Sleep Apnea | Interrupted breathing during sleep |

The application also provides **health suggestions based on prediction results**.

---

## Future Improvements

* Deploy application online
* Add real-time sleep tracking
* Integrate wearable health devices
* Improve model accuracy with more data
* Add deep learning model support

---

## Author

**Ravi Kumar**

GitHub:
https://github.com/Ravi-99-alt

---

## ⭐ If you like this project

Give it a **star ⭐ on GitHub**.
