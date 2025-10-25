import os
import cv2
import mediapipe as mp
import pandas as pd
import uuid
import numpy as np
import json
import tensorflow as tf
from collections import deque, Counter
import time
import base64
from PIL import Image
import io
import eventlet
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_socketio import SocketIO, emit
from functools import wraps
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import relationship

# --- App and Database Configuration ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads', 'profiles')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)
# Note: You may need to add async_mode='eventlet' if you haven't already
# for background tasks to work reliably with gunicorn/nginx.
socketio = SocketIO(app)

# --- Database Model Definitions ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(100), nullable=True)
    middlename = db.Column(db.String(100), nullable=True)
    lastname = db.Column(db.String(100), nullable=True)
    phone_num = db.Column(db.String(20), nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    gym_name = db.Column(db.String(100), nullable=False)
    photo_url = db.Column(db.String(200), nullable=True)
    status = db.Column(db.String(20), nullable=False, default='inactive')
    assignments = db.relationship('Assignment', foreign_keys='Assignment.trainer_id', backref='trainer', lazy=True, cascade="all, delete-orphan")
    workout_sessions = db.relationship('WorkoutSession', backref='user', lazy=True, cascade="all, delete-orphan")

class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trainer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assigned_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class WorkoutSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    total_reps = db.Column(db.Integer, default=0)
    error_logs = db.relationship('ErrorLog', backref='session', lazy=True, cascade="all, delete-orphan")

class ErrorLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('workout_session.id'), nullable=False)
    exercise_name = db.Column(db.String(100), nullable=False)
    rep_number = db.Column(db.Integer, nullable=False)
    error_type = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# --- AI MODEL AND STATE INITIALIZATION ---
TRAINING_ARTIFACTS_DIR = 'training'
MODEL_FILENAME = os.path.join(TRAINING_ARTIFACTS_DIR, 'exercise_classifier_lstm.h5')
LABEL_MAPPING_FILENAME = os.path.join(TRAINING_ARTIFACTS_DIR, 'label_mapping.json')
SEQUENCE_LENGTH = 90
CONF_THRESHOLD = 0.80
STABILITY_FRAMES = 10
CLASSIFICATION_INTERVAL = 0.5  # (in seconds) How often to run the heavy AI model (0.5 = 2 times per sec)

try:
    model = tf.keras.models.load_model(MODEL_FILENAME)
    with open(LABEL_MAPPING_FILENAME, 'r') as f:
        label_mapping = {int(k): v for k, v in json.load(f).items()}
    print("✅ AI Model and label mapping loaded.")
except Exception as e:
    print(f"❌ Error loading AI model or labels: {e}")
    model = None

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y, a.z]); b = np.array([b.x, b.y, b.z]); c = np.array([c.x, c.y, c.z])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def extract_angle_features_for_model(landmarks):
    lm = landmarks; mp_pose = mp.solutions.pose.PoseLandmark
    return np.array([
        calculate_angle(lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_ELBOW], lm[mp_pose.LEFT_WRIST]),
        calculate_angle(lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_ELBOW], lm[mp_pose.RIGHT_WRIST]),
        calculate_angle(lm[mp_pose.LEFT_ELBOW], lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_HIP]),
        calculate_angle(lm[mp_pose.RIGHT_ELBOW], lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_HIP]),
        calculate_angle(lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_HIP], lm[mp_pose.LEFT_KNEE]),
        calculate_angle(lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_HIP], lm[mp_pose.RIGHT_KNEE]),
        calculate_angle(lm[mp_pose.LEFT_HIP], lm[mp_pose.LEFT_KNEE], lm[mp_pose.LEFT_ANKLE]),
        calculate_angle(lm[mp_pose.RIGHT_HIP], lm[mp_pose.RIGHT_KNEE], lm[mp_pose.RIGHT_ANKLE])
    ])

def calculate_angle_2d(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

class ExerciseAnalyzer:
    def __init__(self, reset_timeout=5.0):
        self.rep_counter = 0; self.stage = None; self.form_status = "START EXERCISE"
        self.status_color = (0, 255, 0); self.previous_exercise = "neutral"
        self.last_rep_time = time.time(); self.RESET_TIMEOUT = reset_timeout
        self.session_active = False; self.error_log = []
        self.consecutive_error_counter = 0
        self.last_consecutive_error_type = None
        self.triggered_alert = None

    def get_triggered_alert(self):
        """ A method to fetch and clear a pending alert. """
        try:
            return self.triggered_alert
        finally:
            self.triggered_alert = None

    def analyze_frame(self, exercise_name, landmarks):
        if landmarks is None:
            if exercise_name == "neutral": self.previous_exercise = "neutral"; self.stage = None
            return
            
        if exercise_name != self.previous_exercise:
            self.rep_counter = 0; self.stage = None; self.previous_exercise = exercise_name
            self.last_rep_time = time.time()
            self.consecutive_error_counter = 0
            self.last_consecutive_error_type = None

        self.form_status = "CORRECT FORM"; self.status_color = (0, 255, 0)
        if time.time() - self.last_rep_time > self.RESET_TIMEOUT and self.stage is not None:
            self.stage = None; self.form_status = "INACTIVE - RESET"
        
        mp_lm = mp.solutions.pose.PoseLandmark
        try:
            prev_rep_counter = self.rep_counter
            
            if 'bicepCurl' in exercise_name:
                left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]; left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]; left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]; left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]
                right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]; right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]; right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]; right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]
                elbow_angle, shoulder_angle = 0, 0
                if left_wrist_lm.visibility > right_wrist_lm.visibility:
                    elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
                    shoulder_angle = calculate_angle_2d([left_elbow_lm.x, left_elbow_lm.y], [left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y])
                else:
                    elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])
                    shoulder_angle = calculate_angle_2d([right_elbow_lm.x, right_elbow_lm.y], [right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y])
                if elbow_angle > 160: self.stage = "down"
                if elbow_angle < 70 and self.stage == 'down': self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                if shoulder_angle > 55: self.form_status = "ERROR: KEEP ELBOWS PINNED"; self.status_color = (0, 0, 255)
            elif 'shoulderPress' in exercise_name:
                left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]; left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]; left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]; left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]; left_knee_lm = landmarks[mp_lm.LEFT_KNEE.value]
                right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]; right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]; right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]; right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]; right_knee_lm = landmarks[mp_lm.RIGHT_KNEE.value]
                left_elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
                right_elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])
                avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
                if avg_elbow_angle < 100: self.stage = "down"
                if avg_elbow_angle > 160 and self.stage == 'down': self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                if self.stage == 'down':
                    torso_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y], [left_knee_lm.x, left_knee_lm.y])
                    if torso_angle < 70 or torso_angle > 100: self.form_status = "ERROR: RECLINE AT 85 DEGREES"; self.status_color = (0, 0, 255)
                    left_shoulder_angle = calculate_angle_2d([left_elbow_lm.x, left_elbow_lm.y], [left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y])
                    right_shoulder_angle = calculate_angle_2d([right_elbow_lm.x, right_elbow_lm.y], [right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y])
                    avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
                    if avg_shoulder_angle < 25 or avg_shoulder_angle > 65: self.form_status = "WARNING: TUCK ELBOWS AT 45 DEG"; self.status_color = (0, 165, 255)
            elif 'lateralRaise' in exercise_name:
                left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]; left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]; left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]; left_knee_lm = landmarks[mp_lm.LEFT_KNEE.value]
                right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]; right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]; right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]; right_knee_lm = landmarks[mp_lm.RIGHT_KNEE.value]
                shoulder_angle = 0
                if left_elbow_lm.visibility > right_elbow_lm.visibility:
                    shoulder_angle = calculate_angle_2d([left_elbow_lm.x, left_elbow_lm.y], [left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y])
                else:
                    shoulder_angle = calculate_angle_2d([right_elbow_lm.x, right_elbow_lm.y], [right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y])
                if shoulder_angle < 20: self.stage = "down"
                if shoulder_angle > 95: self.form_status = "WARNING: DO NOT OVER-RAISE"; self.status_color = (0, 165, 255)
                if shoulder_angle > 75 and self.stage == 'down': self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                torso_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y], [left_knee_lm.x, left_knee_lm.y])
                if torso_angle < 160 or torso_angle > 175: self.form_status = "ERROR: LEAN FORWARD 10-20 DEG"; self.status_color = (0, 0, 255)
            elif 'tricepKickback' in exercise_name:
                left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]; left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]; left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]; left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]; left_knee_lm = landmarks[mp_lm.LEFT_KNEE.value]
                right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]; right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]; right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]; right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]; right_knee_lm = landmarks[mp_lm.RIGHT_KNEE.value]
                elbow_angle = 0
                if left_wrist_lm.visibility > right_wrist_lm.visibility:
                    elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
                else:
                    elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])
                if elbow_angle < 100: self.stage = "in"
                if elbow_angle > 160 and self.stage == 'in': self.stage = "out"; self.rep_counter += 1; self.last_rep_time = time.time()
                if self.stage is not None:
                    left_torso_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y], [left_knee_lm.x, left_knee_lm.y])
                    right_torso_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y], [right_knee_lm.x, right_knee_lm.y])
                    torso_angle = left_torso_angle if left_hip_lm.visibility > right_hip_lm.visibility else right_torso_angle
                    if torso_angle > 145: self.form_status = "ERROR: BEND OVER MORE"; self.status_color = (0, 0, 255)
                    elif self.stage == 'out' and elbow_angle < 155: self.form_status = "EXTEND ARM FULLY"; self.status_color = (0, 165, 255)
            elif 'bentOverRow' in exercise_name:
                left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]; left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]; left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]; left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]; left_knee_lm = landmarks[mp_lm.LEFT_KNEE.value]
                right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]; right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]; right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]; right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]; right_knee_lm = landmarks[mp_lm.RIGHT_KNEE.value]
                elbow_angle = 0
                if left_wrist_lm.visibility > right_wrist_lm.visibility:
                    elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
                else:
                    elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])
                if elbow_angle > 150: self.stage = "down"
                if elbow_angle < 80 and self.stage == 'down': self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                if self.stage is not None:
                    left_torso_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y], [left_knee_lm.x, left_knee_lm.y])
                    right_torso_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y], [right_knee_lm.x, right_knee_lm.y])
                    torso_angle = left_torso_angle if left_hip_lm.visibility > right_hip_lm.visibility else right_torso_angle
                    if torso_angle > 145: self.form_status = "ERROR: BEND OVER MORE"; self.status_color = (0, 0, 255)
            
            is_new_rep = (self.rep_counter > prev_rep_counter)

            if is_new_rep:
                if "ERROR" in self.form_status:
                    if self.session_active:
                        log_entry = { "rep_number": self.rep_counter, "error_type": self.form_status, "exercise_name": exercise_name }
                        self.error_log.append(log_entry)
                        print(f"LOGGED ERROR: {log_entry}")
                    if self.form_status == self.last_consecutive_error_type:
                        self.consecutive_error_counter += 1
                    else:
                        self.last_consecutive_error_type = self.form_status
                        self.consecutive_error_counter = 1
                else:
                    self.consecutive_error_counter = 0
                    self.last_consecutive_error_type = None

                if self.consecutive_error_counter >= 6:
                    alert_message = f"Repeated Error: {self.last_consecutive_error_type.replace('ERROR: ', '')}"
                    self.triggered_alert = {'message': alert_message, 'exercise': exercise_name, 'reps': self.rep_counter}
                    print(f"!!! TRAINER ALERT TRIGGERED: {alert_message} !!!")
                    self.consecutive_error_counter = 0
        except Exception as e:
            print(f"Error during form analysis: {e}")
            self.form_status = "ERROR: TRACKING LOST"; self.status_color = (0,0,255)

    def get_status(self):
        return self.rep_counter, self.form_status, self.status_color
    
    def reset_session(self):
        self.rep_counter = 0; self.stage = None; self.error_log = []
        self.session_active = True
        self.consecutive_error_counter = 0
        self.last_consecutive_error_type = None
        self.triggered_alert = None
        print("Session started. Logging and alert tracking enabled.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
clients = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('user_role') != 'Gym Owner':
            flash('You do not have permission to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- (All your Flask routes: @app.route(...)) ---
# --- (No changes are needed to any of your regular Flask routes) ---
# --- (e.g., home, register, login, dashboard, profile, admin_dashboard, etc.) ---
# --- (These are all left exactly as you wrote them) ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        if User.query.filter_by(email=email).first():
            flash('Email address already registered.', 'error')
            return redirect(url_for('register'))
        hashed_password = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
        new_user = User(
            firstname=request.form.get('firstname'), middlename=request.form.get('middlename'),
            lastname=request.form.get('lastname'), phone_num=request.form.get('phoneNum'),
            gender=request.form.get('gender'), email=email, password_hash=hashed_password,
            gym_name=request.form.get('gymName'), role='Gym Owner', status='active'
        )
        db.session.add(new_user); db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            session['user_id'] = user.id; session['user_role'] = user.role
            session['user_gym_name'] = user.gym_name; session['user_firstname'] = user.firstname
            session['user_lastname'] = user.lastname
            return redirect(url_for('admin_dashboard') if user.role == 'Gym Owner' else url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route("/dashboard")
@login_required
def dashboard(): return render_template("dashboard.html")

@app.route("/monitor")
@login_required
def monitor(): return render_template("monitor.html")

@app.route("/settings")
@login_required
def settings(): return render_template("settings.html")

@app.route("/security_settings")
@login_required
def security_settings():
    return render_template("security_settings.html")

@app.route("/delete_account")
@login_required
def delete_account():
    return render_template("delete_account.html")

@app.route("/profile", methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))
    if request.method == 'POST':
        user.firstname = request.form.get('firstname'); user.lastname = request.form.get('lastname')
        user.email = request.form.get('email'); user.phone_num = request.form.get('phone_num')
        if 'gym_name' in request.form: user.gym_name = request.form.get('gym_name')
        if 'photo' in request.files:
            file = request.files['photo']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(f"{user.id}_{file.filename}")
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                user.photo_url = os.path.join('uploads/profiles', filename).replace('\\', '/')
        db.session.commit()
        session['user_firstname'] = user.firstname; session['user_lastname'] = user.lastname
        if 'gym_name' in request.form: session['user_gym_name'] = user.gym_name
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    return render_template("profile.html", user=user)

@app.route("/change_password", methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        data = request.get_json()
        user = User.query.get(session.get('user_id'))
        if not user: return jsonify({'status': 'error', 'message': 'User not found.'}), 404
        if not bcrypt.check_password_hash(user.password_hash, data.get('currentPassword')):
            return jsonify({'status': 'error', 'message': 'Incorrect current password.'}), 400
        hashed_new_password = bcrypt.generate_password_hash(data.get('newPassword')).decode('utf-8')
        user.password_hash = hashed_new_password
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Password updated successfully!'}), 200
    return render_template("change_password.html")

@app.route('/profile/change_password', methods=['POST'])
@login_required
def change_password_submit():
    user = User.query.get(session['user_id'])
    if not bcrypt.check_password_hash(user.password_hash, request.form.get('current_password')):
        flash('Your current password was incorrect. Please try again.', 'error')
    else:
        user.password_hash = bcrypt.generate_password_hash(request.form.get('new_password')).decode('utf-8')
        db.session.commit()
        flash('Your password has been changed successfully!', 'success')
    return redirect(url_for('profile'))

@app.route("/delete-account", methods=['DELETE'])
@login_required
def delete_user_account():
    try:
        user_to_delete = User.query.get(session.get('user_id'))
        if not user_to_delete: return jsonify({'success': False, 'message': 'User not found.'}), 404
        db.session.delete(user_to_delete); db.session.commit()
        session.clear()
        return jsonify({'success': True, 'message': 'Your account has been successfully deleted.'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': 'An error occurred during deletion.'}), 500

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    current_assignments = db.session.query(Assignment).join(User).all()
    today = datetime.utcnow().date()
    start_of_week = today - timedelta(days=today.weekday())
    start_of_month = today.replace(day=1)
    total_errors_today = db.session.query(func.count(ErrorLog.id)).filter(func.date(ErrorLog.timestamp) == today).scalar() or 0
    most_common_error_week_query = db.session.query(ErrorLog.error_type, func.count(ErrorLog.id).label('count')).filter(ErrorLog.timestamp >= start_of_week).group_by(ErrorLog.error_type).order_by(func.count(ErrorLog.id).desc()).first()
    most_common_error_week = most_common_error_week_query[0].replace('ERROR: ', '') if most_common_error_week_query else "N/A"
    total_errors_month = db.session.query(func.count(ErrorLog.id)).filter(ErrorLog.timestamp >= start_of_month).scalar() or 0
    recent_errors = db.session.query(ErrorLog, User).join(WorkoutSession, ErrorLog.session_id == WorkoutSession.id).join(User, WorkoutSession.user_id == User.id).order_by(ErrorLog.timestamp.desc()).limit(5).all()
    muscle_group_mapping = {
        'bicepCurl': 'Arms', 'tricepKickback': 'Arms', 'shoulderPress': 'Arms',
        'lateralRaise': 'Arms', 'bentOverRow': 'Back',
    }
    current_month_chart_data = {'chest': 0, 'back': 0, 'legs': 0, 'arms': 0}
    errors_this_month = db.session.query(ErrorLog.exercise_name, func.count(ErrorLog.id).label('count')).filter(ErrorLog.timestamp >= start_of_month).group_by(ErrorLog.exercise_name).all()
    for exercise, count in errors_this_month:
        group = muscle_group_mapping.get(exercise)
        if group and group in current_month_chart_data:
            current_month_chart_data[group] += count
    return render_template(
        "admin_dashboard.html", assignments=current_assignments,
        total_errors_today=total_errors_today, most_common_error_week=most_common_error_week,
        total_errors_month=total_errors_month, recent_errors=recent_errors,
        current_month_chart_data=json.dumps(current_month_chart_data)
    )

@app.route("/admin/analytics/<int:user_id>")
@admin_required
def analytics(user_id):
    user = User.query.get_or_404(user_id)
    sessions = WorkoutSession.query.filter_by(user_id=user.id).order_by(WorkoutSession.start_time.desc()).all()
    all_errors = [error.error_type for s in sessions for error in s.error_logs]
    error_counts = Counter(all_errors)
    most_common_errors = error_counts.most_common(5)
    return render_template("analytics.html", user=user, sessions=sessions, most_common_errors=most_common_errors)

@app.route("/admin/trainers")
@admin_required
def trainers():
    all_trainers = User.query.filter_by(role='Trainer').all()
    assigned_trainer_ids = [a.trainer_id for a in Assignment.query.all()]
    return render_template("trainers.html", trainers=all_trainers, assigned_trainer_ids=assigned_trainer_ids)

@app.route("/admin/edit_gym_name", methods=['GET', 'POST'])
@admin_required
def admin_edit_gym_name():
    if request.method == 'POST':
        data = request.get_json()
        new_gym_name = data.get('new_gym_name')
        if not new_gym_name: return jsonify({'status': 'error', 'message': 'New gym name is required.'}), 400
        user = User.query.get(session.get('user_id'))
        if user:
            user.gym_name = new_gym_name; db.session.commit()
            session['user_gym_name'] = new_gym_name
            return jsonify({'status': 'success', 'message': 'Gym name updated successfully!'})
        else:
            return jsonify({'status': 'error', 'message': 'User not found.'}), 404
    return render_template("admin_edit_gym_name.html")

@app.route("/admin/trainers/add", methods=['POST'])
@admin_required
def add_trainer():
    email = request.form.get('email')
    if User.query.filter_by(email=email).first(): return jsonify({'status': 'error', 'message': 'Email already exists.'})
    hashed_password = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
    new_trainer = User(
        firstname=request.form.get('firstname'), lastname=request.form.get('lastname'),
        email=email, phone_num=request.form.get('phone'), password_hash=hashed_password,
        role='Trainer', gender='N/A', gym_name=session.get('user_gym_name', 'Default Gym'), status='inactive'
    )
    if 'photo' in request.files:
        file = request.files['photo']
        if file and file.filename != '' and allowed_file(file.filename):
            db.session.add(new_trainer); db.session.flush()
            filename = secure_filename(f"{new_trainer.id}_{file.filename}")
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new_trainer.photo_url = os.path.join('uploads/profiles', filename).replace('\\', '/')
    db.session.add(new_trainer); db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer added successfully!'})

@app.route("/admin/trainers/edit/<int:user_id>", methods=['POST'])
@admin_required
def edit_trainer(user_id):
    trainer = User.query.get_or_404(user_id)
    trainer.firstname = request.form.get('firstname'); trainer.lastname = request.form.get('lastname')
    trainer.email = request.form.get('email'); trainer.phone_num = request.form.get('phone')
    if 'photo' in request.files:
        file = request.files['photo']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(f"{trainer.id}_{file.filename}")
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            trainer.photo_url = os.path.join('uploads/profiles', filename).replace('\\', '/')
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer updated successfully!'})

@app.route("/admin/trainers/delete/<int:user_id>", methods=['POST'])
@admin_required
def delete_trainer(user_id):
    trainer = User.query.get_or_404(user_id)
    if trainer.photo_url:
        try: os.remove(os.path.join('static', trainer.photo_url))
        except OSError as e: print(f"Error deleting file: {e.strerror}")
    db.session.delete(trainer); db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer deleted successfully.'})

@app.route("/admin/assign/<int:trainer_id>", methods=['POST'])
@admin_required
def assign_trainer(trainer_id):
    if Assignment.query.filter_by(trainer_id=trainer_id).first(): return jsonify({'status': 'info', 'message': 'This trainer is already assigned.'})
    trainer = User.query.get(trainer_id)
    if trainer:
        assignment = Assignment(trainer_id=trainer_id)
        db.session.add(assignment)
        trainer.status = 'active'
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Trainer assigned to dashboard!'})
    return jsonify({'status': 'error', 'message': 'Trainer not found.'})

@app.route("/admin/unassign/<int:assignment_id>", methods=['POST'])
@admin_required
def unassign_trainer(assignment_id):
    assignment = Assignment.query.get_or_404(assignment_id)
    trainer = User.query.get(assignment.trainer_id)
    if trainer: trainer.status = 'inactive'
    db.session.delete(assignment); db.session.commit()
    flash('Trainer has been un-assigned from the dashboard.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/unassign_by_trainer/<int:trainer_id>', methods=['POST'])
@admin_required
def unassign_by_trainer_id(trainer_id):
    assignment = Assignment.query.filter_by(trainer_id=trainer_id).first()
    if assignment:
        trainer = User.query.get(trainer_id)
        if trainer: trainer.status = 'inactive'
        db.session.delete(assignment); db.session.commit()
        return jsonify({'status': 'success', 'message': 'Trainer unassigned successfully.'})
    return jsonify({'status': 'error', 'message': 'Trainer was not assigned.'})
    
# --- (End of regular Flask routes) ---


# --- START OF OPTIMIZED SOCKETIO CODE ---

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    clients[request.sid] = {
        'camera1': {
            'sequence_buffer': deque(maxlen=SEQUENCE_LENGTH), 
            'prediction_buffer': deque(maxlen=STABILITY_FRAMES), 
            'stable_exercise': 'neutral', 
            'analyzer': ExerciseAnalyzer(),
            'last_form_status': None,
            'is_processing': False,                 # <-- ADDED: Processing lock
            'last_classification_time': 0           # <-- ADDED: Throttling timer
        },
        'camera2': {
            'sequence_buffer': deque(maxlen=SEQUENCE_LENGTH), 
            'prediction_buffer': deque(maxlen=STABILITY_FRAMES), 
            'stable_exercise': 'neutral', 
            'analyzer': ExerciseAnalyzer(),
            'last_form_status': None,
            'is_processing': False,                 # <-- ADDED: Processing lock
            'last_classification_time': 0           # <-- ADDED: Throttling timer
        }
    }

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')
    clients.pop(request.sid, None)

@socketio.on('start_session')
def handle_start_session(data):
    camera_id = data.get('camera_id')
    client_camera_state = clients.get(request.sid, {}).get(camera_id)
    if client_camera_state:
        client_camera_state['analyzer'].reset_session()
        emit('session_started', {'camera_id': camera_id}, room=request.sid)

@socketio.on('end_session')
def handle_end_session(data):
    camera_id = data.get('camera_id')
    client_camera_state = clients.get(request.sid, {}).get(camera_id)
    if not client_camera_state or not client_camera_state['analyzer'].session_active: return
    analyzer = client_camera_state['analyzer']
    analyzer.session_active = False
    if 'user_id' in session and analyzer.rep_counter > 0:
        new_session = WorkoutSession(user_id=session['user_id'], end_time=datetime.utcnow(), total_reps=analyzer.rep_counter)
        db.session.add(new_session); db.session.flush()
        for error in analyzer.error_log:
            new_log = ErrorLog(session_id=new_session.id, exercise_name=error['exercise_name'], rep_number=error['rep_number'], error_type=error['error_type'])
            db.session.add(new_log)
        db.session.commit()
        print(f"Session saved for user {session['user_id']} with {len(analyzer.error_log)} errors.")
        emit('session_saved', {'camera_id': camera_id, 'reps': analyzer.rep_counter, 'errors': len(analyzer.error_log)}, room=request.sid)
    else:
        print("Session ended without saving (no user or no reps).")


def process_frame_task(sid, data):
    """
    This function runs in a background thread and does ALL the heavy AI work.
    It is NOT a direct socketio handler.
    """
    camera_id = None
    client_camera_state = None
    try:
        camera_id = data['camera_id']
        image_b64 = data['image_data']
        client_camera_state = clients.get(sid, {}).get(camera_id)
        if not client_camera_state:
            return # Client disconnected or state lost
            
        sbuf = io.BytesIO(); sbuf.write(base64.b64decode(image_b64.split(',')[1]))
        pimg = Image.open(sbuf); frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); results = pose.process(image_rgb)
        
        stable_exercise = client_camera_state['stable_exercise']
        landmarks_for_js = []
        
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark: landmarks_for_js.append({'x': lm.x, 'y': lm.y, 'visibility': lm.visibility})
            angle_features = extract_angle_features_for_model(results.pose_landmarks.landmark)
            
            if not np.any(np.isnan(angle_features)):
                client_camera_state['sequence_buffer'].append(angle_features)
                
                # --- OPTIMIZATION (Solution 3: AI Throttling) ---
                current_time = time.time()
                if (len(client_camera_state['sequence_buffer']) == SEQUENCE_LENGTH and
                   (current_time - client_camera_state['last_classification_time'] > CLASSIFICATION_INTERVAL)):
                    
                    client_camera_state['last_classification_time'] = current_time # Reset timer
                    
                    input_data = np.expand_dims(np.array(client_camera_state['sequence_buffer']), axis=0)
                    pred_probs = model.predict(input_data, verbose=0)[0]
                    pred_idx = np.argmax(pred_probs)
                    pred_class = label_mapping.get(pred_idx, "unknown")
                    
                    if pred_probs[pred_idx] >= CONF_THRESHOLD:
                        client_camera_state['prediction_buffer'].append(pred_class)
                        if len(client_camera_state['prediction_buffer']) == STABILITY_FRAMES and len(set(client_camera_state['prediction_buffer'])) == 1:
                            stable_exercise = client_camera_state['prediction_buffer'][0]
                    else:
                        client_camera_state['prediction_buffer'].clear(); stable_exercise = "neutral"
            
            # Form analysis runs on *every* processed frame, using the last known exercise
            client_camera_state['analyzer'].analyze_frame(stable_exercise, results.pose_landmarks.landmark)
        else:
            client_camera_state['sequence_buffer'].clear(); client_camera_state['prediction_buffer'].clear()
            stable_exercise = "neutral"; client_camera_state['analyzer'].analyze_frame("neutral", None)
        
        alert_data = client_camera_state['analyzer'].get_triggered_alert()
        if alert_data:
            alert_data['camera_id'] = camera_id
            socketio.emit('trainer_alert', alert_data, room=sid)

        client_camera_state['stable_exercise'] = stable_exercise
        reps, form, color = client_camera_state['analyzer'].get_status()

        last_form = client_camera_state.get('last_form_status')
        if "ERROR" in form and form != last_form:
            message = form.replace('ERROR: ', '')
            socketio.emit('form_error', {'message': message, 'camera_id': camera_id}, room=sid)
            client_camera_state['last_form_status'] = form
        elif "ERROR" not in form:
            client_camera_state['last_form_status'] = None

        socketio.emit('response', {
            'exercise': stable_exercise, 'reps': reps, 'form_status': form,
            'landmarks': landmarks_for_js, 'camera_id': camera_id
        }, room=sid)
        
    except Exception as e:
        print(f"Error processing image for camera {camera_id} (SID: {sid}): {e}")
    finally:
        # --- CRITICAL ---
        # Release the lock so the next frame can be processed
        if client_camera_state:
            client_camera_state['is_processing'] = False


@socketio.on('image')
def handle_image(data):
    """
    This is the NEW handle_image. It is lightweight and returns instantly.
    It just checks the lock and starts the background task.
    """
    if model is None: return
    try:
        sid = request.sid
        camera_id = data['camera_id']
    except (TypeError, KeyError): 
        return # Invalid data

    client_camera_state = clients.get(sid, {}).get(camera_id)
    if not client_camera_state:
        return # No state for this client/camera

    # --- OPTIMIZATION (Solution 2: Frame Skipping) ---
    # If a frame is already being processed, just drop this new one.
    if client_camera_state.get('is_processing', False):
        # print(f"Dropping frame for {camera_id} (already processing)")
        return
        
    # Set the lock *before* starting the task
    client_camera_state['is_processing'] = True
    
    # Start the background task. This call returns immediately.
    # The main server thread is now free to handle other requests.
    socketio.start_background_task(process_frame_task, sid, data)

# --- END OF OPTIMIZED SOCKETIO CODE ---


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print("Starting Flask-SocketIO server...")
    # You might need to use eventlet or gevent for production:
    # socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    # For development, this is fine:
    socketio.run(app, debug=True)