import os
import cv2
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
import re
import mediapipe as mp 
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_socketio import SocketIO, emit, join_room, leave_room
from functools import wraps
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from werkzeug.utils import secure_filename
from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import relationship
from ultralytics import YOLO 
from datetime import timedelta

from analysis_logic import ExerciseAnalyzer 
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from flask_mail import Mail, Message
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')

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
socketio = SocketIO(app, async_mode='eventlet') 

# --- Mail Configuration ---
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() in ('true', '1', 't')
app.config['MAIL_USE_SSL'] = os.environ.get('MAIL_USE_SSL', 'False').lower() in ('true', '1', 't')
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'your_email@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your_app_password') 
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@yourgymapp.com')
app.config['SECURITY_EMAIL_SALT'] = 'email-confirm-salt'

mail = Mail(app)
s = URLSafeTimedSerializer(app.secret_key)

# --- Database Model Definitions ---

class Gym(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    users = db.relationship('User', backref='gym', lazy=True)
    sessions = db.relationship('WorkoutSession', backref='gym', lazy=True)

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
    gym_id = db.Column(db.Integer, db.ForeignKey('gym.id'), nullable=False)
    photo_url = db.Column(db.String(200), nullable=True)
    status = db.Column(db.String(20), nullable=False, default='unverified')
    assignments = db.relationship('Assignment', foreign_keys='Assignment.trainer_id', backref='trainer', lazy=True, cascade="all, delete-orphan")
    workout_sessions = db.relationship('WorkoutSession', backref='user', lazy=True, cascade="all, delete-orphan")
    
    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trainer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assigned_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class WorkoutSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    gym_id = db.Column(db.Integer, db.ForeignKey('gym.id'), nullable=False)
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
SEQUENCE_LENGTH = 90
CONF_THRESHOLD = 0.30 
STABILITY_FRAMES = 10
TRAINING_ARTIFACTS_DIR = os.path.join(basedir, 'training') 

interpreter = None 
scaler = None 
input_details = None
output_details = None
label_mapping = {}

# Initialize YOLO for TRACKING only
yolo_model = YOLO('yolov8n-pose.pt')
YOLO_CONF_THRESHOLD = 0.60

# Mapping for Fallback (YOLO index -> MediaPipe index)
YOLO_TO_MP = {
    0: 0,   # nose
    5: 11,  # left_shoulder
    6: 12,  # right_shoulder
    7: 13,  # left_elbow
    8: 14,  # right_elbow
    9: 15,  # left_wrist
    10: 16, # right_wrist
    11: 23, # left_hip
    12: 24, # right_hip
    13: 25, # left_knee
    14: 26, # right_knee
    15: 27, # left_ankle
    16: 28  # right_ankle
}

# --- Initialize MediaPipe for LANDMARKS ---
mp_pose = mp.solutions.pose
pose_extractor = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def load_model_and_labels():
    global interpreter, label_mapping, input_details, output_details, scaler
    try:
        TFLITE_MODEL_FILENAME = os.path.join(TRAINING_ARTIFACTS_DIR, 'exercise_classifier_quant.tflite')
        LABEL_MAPPING_FILENAME = os.path.join(TRAINING_ARTIFACTS_DIR, 'label_mapping.json')
        SCALER_FILENAME = os.path.join(TRAINING_ARTIFACTS_DIR, 'scaler.pkl')
        
        if os.path.exists(TFLITE_MODEL_FILENAME):
            interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_FILENAME)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("✅ TFLite Interpreter loaded successfully.")
        else:
            print(f"❌ Error: TFLite model file not found at {TFLITE_MODEL_FILENAME}")

        if os.path.exists(LABEL_MAPPING_FILENAME):
            with open(LABEL_MAPPING_FILENAME, 'r') as f:
                label_mapping = {int(k): v for k, v in json.load(f).items()}
            print("✅ Label mapping loaded successfully.")
        else:
            print(f"❌ Error: Label mapping file not found.")

        # LOAD SCALER
        if os.path.exists(SCALER_FILENAME):
            scaler = joblib.load(SCALER_FILENAME)
            print("✅ Scaler loaded successfully.")
        else:
            print(f"❌ Error: Scaler file not found at {SCALER_FILENAME}. Model will fail to predict correctly!")
            
    except Exception as e:
        print(f"❌ Error loading model/scaler/labels: {e}") 

load_model_and_labels()

# --- Utility Functions ---

def generate_verification_token(email):
    return s.dumps(email, salt=app.config['SECURITY_EMAIL_SALT'])

def confirm_verification_token(token, expiration=3600):
    try:
        email = s.loads(token, salt=app.config['SECURITY_EMAIL_SALT'], max_age=expiration)
        return email
    except SignatureExpired:
        return None
    except Exception:
        return None

def send_verification_email(user_email, token):
    verify_url = url_for('verify_account', token=token, _external=True)
    msg = Message(
        subject="Confirm Your Gym Buddy Account",
        recipients=[user_email],
        html=f"""
            <p>Welcome to Gym Buddy! Please click the link below to verify your email:</p>
            <p><a href="{verify_url}">Verify My Email</a></p>
        """
    )
    mail.send(msg)
    
def format_exercise_name(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).capitalize()

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

# --- Routes ---
@app.context_processor
def utility_processor():
    def to_gmt8(utc_dt):
        if not utc_dt:
            return ""
        # Manually adds 8 hours to the UTC time stored in DB
        return utc_dt + timedelta(hours=8)
    return dict(to_gmt8=to_gmt8)

@app.route("/")
def home():
    if 'user_id' in session:
        if session.get('user_role') == 'Gym Owner':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    return render_template("index.html")

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        firstname = request.form.get('firstname')
        middlename = request.form.get('middlename')
        lastname = request.form.get('lastname')
        phone_num = request.form.get('phoneNum')
        gender = request.form.get('gender')
        email = request.form.get('email')
        password = request.form.get('password')
        gym_name = request.form.get('gymName')
        
        if not all([firstname, lastname, email, password, gym_name]):
            flash('Please fill out all required fields.', 'danger')
            return redirect(url_for('register'))

        gym = Gym.query.filter_by(name=gym_name).first()
        if not gym:
            gym = Gym(name=gym_name)
            db.session.add(gym)
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                flash(f"Error creating new gym: {e}", "danger")
                return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email address already registered.', 'error')
            return redirect(url_for('register'))
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        verification_token = generate_verification_token(email)
        
        new_user = User(
            firstname=firstname, middlename=middlename, lastname=lastname,
            phone_num=phone_num, gender=gender, email=email, 
            password_hash=hashed_password, 
            role='Gym Owner', 
            status='unverified',
            gym_id=gym.id 
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        try:
            send_verification_email(email, verification_token) 
            flash('Registration successful! Please check your email to verify your account.', 'success')
        except Exception as e:
            print(f"ERROR SENDING EMAIL: {e}")
            flash('Registration successful, but we could not send the verification email.', 'warning')
        
        return redirect(url_for("verify_message")) 
    return render_template("register.html")

@app.route("/verify/<string:token>")
def verify_account(token):
    email = confirm_verification_token(token)
    if not email:
        flash('The verification link is invalid or has expired.', 'error')
        return redirect(url_for('login'))
        
    user = User.query.filter_by(email=email).first()
    if user:
        if user.status == 'active':
            flash('Your account is already verified. Please log in.', 'success')
        else:
            user.status = 'active'
            db.session.commit()
            flash('Email verified!', 'success')
    else:
        flash('Account not found.', 'error') 
        
    return redirect(url_for('login'))

@app.route("/verify_message")
def verify_message():
    return render_template("verify_message.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            if user.status == 'unverified':
                 flash('Your account is not verified. Please check your email.', 'error')
                 return redirect(url_for('login'))
            
            session.clear() 
            session['user_id'] = user.id
            session['user_role'] = user.role
            session['user_gym_id'] = user.gym_id
            session['user_gym_name'] = user.gym.name 
            session['user_firstname'] = user.firstname
            session['user_lastname'] = user.lastname
            session['user_email'] = user.email 
            session['user_phone_num'] = user.phone_num 
            session['user_photo_url'] = user.photo_url if user.photo_url else 'src/images/Default_pfp.jpg'
            
            return redirect(url_for('admin_dashboard') if user.role == 'Gym Owner' else url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')
            
    return render_template("login.html")

@app.route("/resend_verification", methods=['GET', 'POST'])
def resend_verification():
    if request.method == 'POST':
        email = request.form.get('email')
        if not email:
            flash('Please enter your email address.', 'warning')
            return redirect(url_for('resend_verification'))

        user = User.query.filter_by(email=email, status='unverified').first()
        if user:
            new_token = generate_verification_token(email)
            try:
                send_verification_email(email, new_token)
                flash('A new verification link has been sent to your email.', 'success')
            except Exception as e:
                print(f"ERROR RESENDING EMAIL: {e}")
                flash('Could not resend the verification email.', 'warning')
        else:
            flash('If an unverified account exists, a new link has been sent.', 'info')
        return redirect(url_for('login'))
    return render_template("resend_form.html")

@app.route("/logout")
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route("/dashboard")
@login_required
def dashboard(): 
    user_id = session['user_id']
    today = datetime.utcnow().date()
    start_of_week = today - timedelta(days=today.weekday())
    start_of_month = today.replace(day=1)
    
    total_errors_today = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.user_id == user_id, func.date(ErrorLog.timestamp) == today).scalar() or 0
    most_common_error_week_query = db.session.query(ErrorLog.error_type, func.count(ErrorLog.id).label('count')).join(WorkoutSession).filter(WorkoutSession.user_id == user_id, ErrorLog.timestamp >= start_of_week).group_by(ErrorLog.error_type).order_by(func.count(ErrorLog.id).desc()).first()
    most_common_error_week = most_common_error_week_query[0].replace('ERROR: ', '') if most_common_error_week_query else "N/A"
    total_errors_month = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.user_id == user_id, ErrorLog.timestamp >= start_of_month).scalar() or 0
    recent_errors = db.session.query(ErrorLog, User).join(WorkoutSession, ErrorLog.session_id == WorkoutSession.id).join(User, WorkoutSession.user_id == User.id).filter(User.id == user_id).order_by(ErrorLog.timestamp.desc()).limit(5).all()

    muscle_group_mapping = {
        'bicepCurl': 'arms', 'tricepKickback': 'arms', 'shoulderPress': 'arms',
        'lateralRaise': 'arms', 'bentOverRow': 'back',
    }
    current_month_chart_data = {'chest': 0, 'back': 0, 'legs': 0, 'arms': 0}
    errors_this_month = db.session.query(ErrorLog.exercise_name, func.count(ErrorLog.id).label('count')).join(WorkoutSession).filter(WorkoutSession.user_id == user_id, ErrorLog.timestamp >= start_of_month).group_by(ErrorLog.exercise_name).all()

    for exercise, count in errors_this_month:
        group = muscle_group_mapping.get(exercise)
        if group and group in current_month_chart_data:
            current_month_chart_data[group] += count
            
    all_sessions = WorkoutSession.query.filter_by(user_id=user_id).filter(WorkoutSession.end_time != None).order_by(WorkoutSession.start_time.desc()).all()

    return render_template(
        "dashboard.html", 
        total_errors_today=total_errors_today,
        most_common_error_week=most_common_error_week,
        total_errors_month=total_errors_month,
        recent_errors=recent_errors,
        current_month_chart_data=current_month_chart_data,
        sessions=all_sessions 
    )

@app.route("/my_sessions")
@login_required
def my_sessions():
    if session.get('user_role') != 'Trainer':
        flash('This page is for trainers.', 'error')
        return redirect(url_for('admin_dashboard'))

    user_id = session['user_id']
    gym_id = session['user_gym_id']
    
    trainer = User.query.get(user_id)
    if not trainer:
        flash('User not found.', 'error')
        return redirect(url_for('dashboard'))

    all_sessions = WorkoutSession.query.filter_by(gym_id=gym_id, user_id=user_id).filter(WorkoutSession.end_time != None).order_by(WorkoutSession.start_time.desc()).all()
    
    return render_template("trainer_session_log.html", sessions=all_sessions, trainer=trainer)

@app.route("/monitor")
@login_required
def monitor(): 
    return render_template("monitor.html")

@app.route("/errorlogpage")
@login_required
def errorlogpage(): 
    gym_id = session['user_gym_id']
    all_errors_query = db.session.query(ErrorLog, User).join(WorkoutSession, ErrorLog.session_id == WorkoutSession.id).join(User, WorkoutSession.user_id == User.id).filter(WorkoutSession.gym_id == gym_id).order_by(ErrorLog.timestamp.desc()).all()

    js_errors = []
    for log, user in all_errors_query:
        js_errors.append({
            'id': log.id,
            'userName': f"{user.firstname} {user.lastname}",
            'userPhoto': url_for('static', filename=user.photo_url if user.photo_url else 'src/images/Default_pfp.jpg'),
            'errorType': log.error_type.replace('ERROR: ', ''),
            'exerciseName': format_exercise_name(log.exercise_name),
            'timeOfError': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'month': log.timestamp.strftime('%b') 
        })

    return render_template("errorlogpage.html", all_errors_json=json.dumps(js_errors))

@app.route("/settings")
@login_required
def settings(): 
    return render_template("settings.html")

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
        user.firstname = request.form.get('firstname')
        user.lastname = request.form.get('lastname')
        user.email = request.form.get('email')
        user.phone_num = request.form.get('phone_num')
        user.gender = request.form.get('gender') 
        
        if 'photo' in request.files:
            file = request.files['photo']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(f"{user.id}_{file.filename}")
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
                new_photo_url = os.path.join('uploads/profiles', filename).replace('\\', '/')
                user.photo_url = new_photo_url
                session['user_photo_url'] = new_photo_url
        
        if 'gym_name' in request.form and session.get('user_role') == 'Gym Owner': 
            gym = Gym.query.get(user.gym_id)
            if gym:
                gym.name = request.form.get('gym_name')
                session['user_gym_name'] = gym.name
        
        db.session.commit()
        session['user_firstname'] = user.firstname
        session['user_lastname'] = user.lastname
        session['user_email'] = user.email
        session['user_phone_num'] = user.phone_num
        session['user_gender'] = user.gender

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

# --- ADMIN ROUTES ---
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    gym_id = session['user_gym_id']
    # DEBUG PRINT to check if Gym ID matches the user data
    print(f"DEBUG: Admin Dashboard loading for Gym ID: {gym_id}")

    today = datetime.utcnow().date()
    
    start_of_current_month = today.replace(day=1)
    start_of_last_month = start_of_current_month - relativedelta(months=1)
    
    errors_current_month = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= start_of_current_month).scalar() or 0
    errors_last_month = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= start_of_last_month, ErrorLog.timestamp < start_of_current_month).scalar() or 0

    error_rate_change = 0
    error_rate_color = "gray" 
    error_rate_status = "Month in Progress"

    if errors_last_month > 0:
        change_percent = ((errors_current_month - errors_last_month) / errors_last_month) * 100
        if change_percent < 0:
            error_rate_color = "green"
            error_rate_status = "Less Error"
            error_rate_change = abs(round(change_percent))
        else:
            error_rate_color = "red"
            error_rate_status = "More Error"
            error_rate_change = abs(round(change_percent))
    elif errors_last_month == 0 and errors_current_month > 0:
        error_rate_color = "red"
        error_rate_status = "More Error"
        error_rate_change = 100 
    elif errors_last_month == 0 and errors_current_month == 0:
         error_rate_color = "gray"
         error_rate_status = "No Errors"
         error_rate_change = 0

    start_of_week = today - timedelta(days=today.weekday())
    current_assignment = db.session.query(Assignment).join(User).filter(User.gym_id == gym_id).first()
    
    last_24_hours = datetime.utcnow() - timedelta(hours=24)
    total_errors_today = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= last_24_hours).scalar() or 0
    
    most_common_error_week_query = db.session.query(ErrorLog.error_type, func.count(ErrorLog.id).label('count')).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= start_of_week).group_by(ErrorLog.error_type).order_by(func.count(ErrorLog.id).desc()).first()
    most_common_error_week = most_common_error_week_query[0].replace('ERROR: ', '') if most_common_error_week_query else "N/A"
    total_errors_month = errors_current_month 
    
    # 1. FETCH RECENT ERROR LOGS
    recent_errors = db.session.query(ErrorLog, User).join(WorkoutSession, ErrorLog.session_id == WorkoutSession.id).join(User, WorkoutSession.user_id == User.id).filter(WorkoutSession.gym_id == gym_id).order_by(ErrorLog.timestamp.desc()).limit(5).all()

    # 2. FETCH IMMEDIATE ACTIONS (CRITICAL ERRORS)
    # We define "Immediate Action" as "Repeated Errors" (based on your logic)
    initial_critical_errors = db.session.query(ErrorLog, User)\
        .join(WorkoutSession, ErrorLog.session_id == WorkoutSession.id)\
        .join(User, WorkoutSession.user_id == User.id)\
        .filter(WorkoutSession.gym_id == gym_id)\
        .filter(ErrorLog.error_type.contains("Repeated Error"))\
        .order_by(ErrorLog.timestamp.desc())\
        .limit(5)\
        .all()

    muscle_group_mapping = {
        'bicepCurl': 'arms', 'tricepKickback': 'arms', 'shoulderPress': 'arms',
        'lateralRaise': 'arms', 'bentOverRow': 'back',
    }
    current_month_chart_data = {'chest': 0, 'back': 0, 'legs': 0, 'arms': 0}
    errors_this_month = db.session.query(ErrorLog.exercise_name, func.count(ErrorLog.id).label('count')).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= start_of_current_month).group_by(ErrorLog.exercise_name).all()
    
    for exercise, count in errors_this_month:
        group = muscle_group_mapping.get(exercise)
        if group and group in current_month_chart_data:
            current_month_chart_data[group] += count

    return render_template(
        "admin_dashboard.html", 
        assignment=current_assignment, 
        total_errors_today=total_errors_today, 
        most_common_error_week=most_common_error_week, 
        total_errors_month=total_errors_month, 
        recent_errors=recent_errors,
        current_month_chart_data=current_month_chart_data,
        error_rate_change=error_rate_change,
        error_rate_color=error_rate_color,
        error_rate_status=error_rate_status,
        critical_errors=initial_critical_errors # <-- Pass Critical Errors here
    )

@app.route("/admin/analytics/<int:user_id>")
@admin_required
def analytics(user_id):
    gym_id = session['user_gym_id']
    user = User.query.filter_by(id=user_id, gym_id=gym_id).first_or_404()
    sessions = WorkoutSession.query.filter_by(user_id=user.id, gym_id=gym_id).order_by(WorkoutSession.start_time.desc()).all()
    all_errors = [error.error_type for s in sessions for error in s.error_logs]
    error_counts = Counter(all_errors)
    most_common_errors = error_counts.most_common(5)
    return render_template("analytics.html", user=user, sessions=sessions, most_common_errors=most_common_errors)

@app.route("/admin/trainers")
@admin_required
def trainers():
    gym_id = session['user_gym_id']
    all_trainers = User.query.filter_by(role='Trainer', gym_id=gym_id).all()
    assigned_trainer_ids = [a.trainer_id for a in Assignment.query.join(User).filter(User.gym_id == gym_id).all()]
    
    last_sessions = {}
    for trainer in all_trainers:
        last_session = WorkoutSession.query.filter_by(user_id=trainer.id, gym_id=gym_id).filter(WorkoutSession.end_time != None).order_by(WorkoutSession.start_time.desc()).first()
        if last_session:
            last_sessions[trainer.id] = last_session

    return render_template("trainers.html", trainers=all_trainers, assigned_trainer_ids=assigned_trainer_ids, last_sessions=last_sessions)

@app.route("/admin/edit_gym_name", methods=['GET', 'POST'])
@admin_required
def admin_edit_gym_name():
    if request.method == 'POST':
        data = request.get_json()
        new_gym_name = data.get('new_gym_name')
        if not new_gym_name: return jsonify({'status': 'error', 'message': 'New gym name is required.'}), 400
        
        gym = Gym.query.get(session['user_gym_id'])
        if gym:
            gym.name = new_gym_name; db.session.commit()
            session['user_gym_name'] = new_gym_name
            return jsonify({'status': 'success', 'message': 'Gym name updated successfully!'})
        else:
            return jsonify({'status': 'error', 'message': 'Gym not found.'}), 404
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
        role='Trainer', gender='N/A', status='inactive', gym_id=session['user_gym_id'] 
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
    trainer = User.query.filter_by(id=user_id, gym_id=session['user_gym_id']).first_or_404()
    
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
    trainer = User.query.filter_by(id=user_id, gym_id=session['user_gym_id']).first_or_404()
    
    if trainer.photo_url:
        try: os.remove(os.path.join('static', trainer.photo_url))
        except OSError as e: print(f"Error deleting file: {e.strerror}")
    db.session.delete(trainer); db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer deleted successfully.'})

@app.route("/admin/assign/<int:trainer_id>", methods=['POST'])
@admin_required
def assign_trainer(trainer_id):
    gym_id = session['user_gym_id'] 
    existing_assignment = db.session.query(Assignment).join(User).filter(User.gym_id == gym_id).first()
    if existing_assignment:
        return jsonify({'status': 'error', 'message': 'A trainer is already assigned. Please unassign them first.'})

    trainer = User.query.filter_by(id=trainer_id, gym_id=gym_id).first()
    if not trainer:
        return jsonify({'status': 'error', 'message': 'Trainer not found.'})

    assignment = Assignment(trainer_id=trainer_id)
    db.session.add(assignment)
    trainer.status = 'active'
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer assigned to dashboard!'})

@app.route("/admin/unassign/<int:assignment_id>", methods=['POST'])
@admin_required
def unassign_trainer(assignment_id):
    assignment = Assignment.query.get_or_404(assignment_id)
    trainer = User.query.get(assignment.trainer_id)
    if not trainer or trainer.gym_id != session['user_gym_id']:
        flash('Permission denied.', 'error')
        return redirect(url_for('admin_dashboard'))
        
    trainer.status = 'inactive'
    db.session.delete(assignment); db.session.commit()
    flash('Trainer has been un-assigned from the dashboard.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/unassign_by_trainer/<int:trainer_id>', methods=['POST'])
@admin_required
def unassign_by_trainer_id(trainer_id):
    trainer = User.query.filter_by(id=trainer_id, gym_id=session['user_gym_id']).first()
    if not trainer:
        return jsonify({'status': 'error', 'message': 'Trainer not found.'})
        
    assignment = Assignment.query.filter_by(trainer_id=trainer_id).first()
    if assignment:
        trainer.status = 'inactive'
        db.session.delete(assignment); db.session.commit()
        return jsonify({'status': 'success', 'message': 'Trainer unassigned successfully.'})
    return jsonify({'status': 'error', 'message': 'Trainer was not assigned.'})

@app.route("/admin/session_log/<int:user_id>")
@admin_required
def trainer_session_log(user_id):
    gym_id = session['user_gym_id']
    trainer = User.query.filter_by(id=user_id, gym_id=gym_id).first_or_404()
    
    all_sessions = WorkoutSession.query.filter_by(gym_id=gym_id, user_id=user_id).filter(WorkoutSession.end_time != None).order_by(WorkoutSession.start_time.desc()).all()
    return render_template("trainer_session_log.html", sessions=all_sessions, trainer=trainer)

@app.route("/session/<int:session_id>")
@login_required
def trainer_session_detail(session_id):
    gym_id = session['user_gym_id']
    session_data = db.session.query(WorkoutSession, User).join(User).filter(WorkoutSession.gym_id == gym_id, WorkoutSession.id == session_id).first_or_404() 
    
    session_obj = session_data[0]
    user_obj = session_data[1]

    duration = "In Progress"
    if session_obj.end_time:
        time_diff = session_obj.end_time - session_obj.start_time
        hours, remainder = divmod(time_diff.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        duration = f"{int(hours)}hrs {int(minutes)}mins"

    errors = ErrorLog.query.filter_by(session_id=session_id).all()
    normal_errors = [e for e in errors if not e.error_type.startswith("Repeated Error")]
    critical_errors = [e for e in errors if e.error_type.startswith("Repeated Error")]

    return render_template(
        "trainer_session.html",
        workout_session=session_obj,
        user=user_obj,
        duration=duration,
        normal_error_count=len(normal_errors),
        critical_error_count=len(critical_errors),
        recent_errors=errors 
    )


# --- SocketIO Handlers ---
clients = {} 

@socketio.on('connect')
def handle_connect():
    if 'user_id' not in session:
        print("Warning: Unauthenticated user tried to connect.")
        return False

    gym_id = session.get('user_gym_id')
    if not gym_id:
        print(f"Warning: User {session.get('user_id')} connected without a gym_id.")
        return False

    room = f'gym_{gym_id}'
    join_room(room)
    clients[request.sid] = {'gym_id': gym_id}
    print(f"Client {request.sid} connected, joined room: {room}")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    client_data = clients.get(sid)

    if client_data:
        gym_id = client_data.get('gym_id')
        
        # --- FIXED: Force save all active cameras for this client ---
        for camera_id in list(client_data.keys()):
            if camera_id not in ['gym_id']: # Skip the gym_id key
                print(f"Force closing session for camera {camera_id} due to disconnect")
                handle_end_session({'camera_id': camera_id, 'sid_for_shutdown': sid})
        # -------------------------------------------------------------

        clients.pop(sid, None) 
        if gym_id:
            room = f'gym_{gym_id}'
            leave_room(room)
            print(f"Client {sid} disconnected, left room: {room}")
    else:
        print(f"Client {sid} disconnected (no data found)")

@socketio.on('start_camera')
def start_camera(data):
    sid = request.sid
    camera_id = data.get('camera_id')
    if not camera_id:
        return

    if sid in clients:
        try:
            clients[sid][camera_id] = {
                'analyzers': {},
                'is_processing': False,
                'active_session_id': None,
                'last_form_status': {}
            }
            print(f"✅ Successfully started camera '{camera_id}' for client {sid}")
        except Exception as e:
            print(f"❌ Error initializing resources for {camera_id}/{sid}: {e}")

@socketio.on('stop_camera')
def stop_camera(data):
    sid = request.sid
    camera_id = data.get('camera_id')
    if not camera_id:
        return

    handle_end_session(data)

    if sid in clients and camera_id in clients[sid]:
        clients[sid].pop(camera_id, None)
        print(f"Stopped camera '{camera_id}' for client {sid}")

@socketio.on('start_session')
def handle_start_session(data):
    camera_id = data.get('camera_id')
    sid = request.sid
    client_camera_state = clients.get(sid, {}).get(camera_id)

    if 'user_id' not in session:
        print(f"Warning: Anonymous user {sid} tried to start session.")
        return

    if client_camera_state:
        for analyzer in client_camera_state['analyzers'].values():
            analyzer.reset_session()

        try:
            new_session = WorkoutSession(
                user_id=session['user_id'],
                gym_id=session['user_gym_id']
            )
            db.session.add(new_session)
            db.session.commit()

            client_camera_state['active_session_id'] = new_session.id
            print(f"Session {new_session.id} started for user {session['user_id']} in gym {session['user_gym_id']}")
            emit('session_started', {'camera_id': camera_id}, room=sid)
        except Exception as e:
            db.session.rollback()
            print(f"Error creating new session in DB: {e}")
    else:
        print(f"Warning: Could not start session. No state found for {sid}/{camera_id}")

@socketio.on('end_session')
def handle_end_session(data):
    camera_id = data.get('camera_id')
    sid = data.get('sid_for_shutdown', request.sid)

    if sid not in clients:
        print(f"Info: handle_end_session called for disconnected client {sid}")
        return

    client_camera_state = clients.get(sid, {}).get(camera_id)
    if not client_camera_state:
        return

    session_id = client_camera_state.get('active_session_id')

    if session_id:
        try:
            with app.app_context():
                session_to_end = WorkoutSession.query.get(session_id)
                if session_to_end:
                    session_to_end.end_time = datetime.utcnow()
                    total_reps = 0
                    for analyzer in client_camera_state['analyzers'].values():
                        total_reps += analyzer.rep_counter

                    session_to_end.total_reps = total_reps
                    db.session.commit()
                    print(f"Session {session_id} ended for user {session_to_end.user_id}. Total Reps: {total_reps}")

                    if not data.get('sid_for_shutdown'):
                        emit('session_saved', {'camera_id': camera_id, 'reps': total_reps}, room=sid)
        except Exception as e:
            db.session.rollback()
            print(f"Error ending session {session_id} in DB: {e}")

    if client_camera_state:
        client_camera_state['active_session_id'] = None
        for analyzer in client_camera_state['analyzers'].values():
            analyzer.reset_session()

def process_frame_task(sid, data, user_info):
    start_time = time.time()
    global interpreter, label_mapping, input_details, output_details, clients, yolo_model, YOLO_CONF_THRESHOLD, pose_extractor, YOLO_TO_MP, scaler

    gym_id = clients.get(sid, {}).get('gym_id')

    try:
        camera_id = data['camera_id']
        client_camera_state = clients.get(sid, {}).get(camera_id)
        if not client_camera_state:
            return
    except (KeyError, TypeError):
        return

    try:
        image_data = base64.b64decode(data['image_data'].split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        frame_rgb = np.array(image)
        frame_rgb.flags.writeable = False
        frame = frame_rgb
    except Exception as e:
        print(f"Error decoding image: {e}")
        if client_camera_state:
            client_camera_state['is_processing'] = False
        return

    # --- YOLOv8 Tracking ---
    try:
        results = yolo_model.track(frame_rgb, verbose=False, conf=YOLO_CONF_THRESHOLD, persist=True)
    except Exception as e:
        print(f"⚠️ YOLO Tracking Error (Skipping Frame): {e}")
        if client_camera_state:
            client_camera_state['is_processing'] = False
        return

    current_frame_data = []

    if results[0].boxes is not None:
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            keypoints = results[0].keypoints

            for i, track_id in enumerate(track_ids):
                if track_id not in client_camera_state['analyzers']:
                    client_camera_state['analyzers'][track_id] = ExerciseAnalyzer(
                        sequence_length=SEQUENCE_LENGTH,
                        conf_threshold=CONF_THRESHOLD,
                        stability_frames=STABILITY_FRAMES
                    )
                    client_camera_state.setdefault('last_form_status', {})[track_id] = None

                analyzer = client_camera_state['analyzers'][track_id]
                final_landmarks_for_ui = [{'x': 0.0, 'y': 0.0, 'z': 0.0, 'visibility': 0.0} for _ in range(33)]

                if keypoints is not None and keypoints.conf is not None:
                    xy = keypoints.xy[i].cpu().numpy()
                    conf = keypoints.conf[i].cpu().numpy()
                    for yolo_idx, mp_idx in YOLO_TO_MP.items():
                        if yolo_idx < len(conf) and conf[yolo_idx] > 0.5:
                            final_landmarks_for_ui[mp_idx] = {
                                'x': float(xy[yolo_idx][0]) / frame.shape[1],
                                'y': float(xy[yolo_idx][1]) / frame.shape[0],
                                'z': 0.0,
                                'visibility': float(conf[yolo_idx])
                            }

                x1, y1, x2, y2 = map(int, boxes[i])
                h, w, _ = frame.shape

                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)

                person_crop = np.ascontiguousarray(frame_rgb[y1:y2, x1:x2])

                if person_crop.size > 0:
                    mp_results = pose_extractor.process(person_crop)

                    if mp_results.pose_landmarks:
                        crop_h, crop_w, _ = person_crop.shape

                        for idx, lm in enumerate(mp_results.pose_landmarks.landmark):
                            px = lm.x * crop_w
                            py = lm.y * crop_h

                            global_px = px + x1
                            global_py = py + y1

                            final_landmarks_for_ui[idx] = {
                                'x': global_px / w,
                                'y': global_py / h,
                                'z': lm.z,
                                'visibility': lm.visibility
                            }

                person_response = {
                    'track_id': track_id,
                    'rep_counter': analyzer.rep_counter,
                    'form_status': analyzer.form_status,
                    'stable_prediction': analyzer.stable_prediction,
                    'landmarks': final_landmarks_for_ui,
                    'debug_angles': {}
                }

                if interpreter and scaler:
                    try:
                        rep_count, form, prediction, angles = analyzer.process_frame(
                            interpreter=interpreter,
                            input_details=input_details,
                            output_details=output_details,
                            label_mapping=label_mapping,
                            landmarks=final_landmarks_for_ui,
                            current_exercise=analyzer.stable_prediction,
                            scaler=scaler
                        )

                        person_response.update({
                            'rep_counter': rep_count,
                            'form_status': form,
                            'stable_prediction': prediction,
                            'debug_angles': {k: int(v) for k, v in angles.items()}
                        })

                        log_entry = analyzer.get_new_error_log()
                        session_id = client_camera_state.get('active_session_id')

                        if log_entry and not session_id:
                            try:
                                with app.app_context():
                                    current_user_id = session.get('user_id')
                                    current_gym_id = gym_id if gym_id else 1

                                    if current_user_id:
                                        new_session = WorkoutSession(
                                            user_id=current_user_id,
                                            gym_id=current_gym_id
                                        )
                                        db.session.add(new_session)
                                        db.session.commit()

                                        session_id = new_session.id
                                        client_camera_state['active_session_id'] = session_id
                                        print(f"✅ Auto-started Session {session_id} for User {current_user_id}")
                                    else:
                                        print("❌ Cannot auto-start session: User ID not found in flask session.")
                            except Exception as e:
                                db.session.rollback()
                                print(f"❌ Failed to auto-start session: {e}")

                        if session_id and log_entry:
                            try:
                                error_type_str = f"[P{track_id}] {log_entry['error_type']}"
                                new_log = ErrorLog(
                                    session_id=session_id,
                                    exercise_name=log_entry['exercise_name'],
                                    rep_number=log_entry['rep_number'],
                                    error_type=error_type_str
                                )
                                db.session.add(new_log)
                                db.session.commit()
                                print(f"📝 Logged Error: {error_type_str}")
                            except Exception as e:
                                db.session.rollback()
                                print(f"Error during logging: {e}")

                        last_form = client_camera_state['last_form_status'].get(track_id)
                        if "ERROR" in form and form != last_form:
                            if gym_id:
                                with app.app_context():
                                    socketio.emit('form_error', {
                                        'message': f"Person {track_id}: {form.replace('ERROR: ', '')}",
                                        'camera_id': camera_id,
                                        'user_name': f"{user_info.get('firstname', 'Unknown')} {user_info.get('lastname', 'User')}",
                                        'timestamp': datetime.utcnow().isoformat()
                                    }, room=f'gym_{gym_id}')
                                    print(f"Emitted 'form_error' to room gym_{gym_id}")
                                client_camera_state['last_form_status'][track_id] = form
                        elif "ERROR" not in form:
                            client_camera_state['last_form_status'][track_id] = None

                        alert_data = analyzer.get_triggered_alert()
                        if alert_data:
                            alert_data['camera_id'] = camera_id
                            alert_data['message'] = f"Person {track_id}: {alert_data['message']}"
                            if gym_id:
                                with app.app_context():
                                    socketio.emit('trainer_alert', alert_data, room=f'gym_{gym_id}')
                                    print(f"Emitted 'trainer_alert' to room gym_{gym_id}: {alert_data}")

                    except Exception as e:
                        print(f"Error analyzing person {track_id}: {e}")

                current_frame_data.append(person_response)

    emit_data = {
        'camera_id': camera_id,
        'people': current_frame_data
    }

    socketio.emit('response', emit_data, room=sid)

    process_time = (time.time() - start_time) * 1000
    if process_time > 100:
        print(f"[DEBUG] Processing took: {process_time:.2f}ms")

    if client_camera_state:
        client_camera_state['is_processing'] = False

@socketio.on('image')
def handle_image(data):
    if interpreter is None:
        return
    try:
        sid = request.sid
        camera_id = data['camera_id']
    except (TypeError, KeyError):
        return

    client_camera_state = clients.get(sid, {}).get(camera_id)
    if not client_camera_state:
        return

    if client_camera_state.get('is_processing', False):
        return

    client_camera_state['is_processing'] = True

    user_info = {
        'firstname': session.get('user_firstname', 'Unknown'),
        'lastname': session.get('user_lastname', 'User')
    }

    socketio.start_background_task(process_frame_task, sid, data, user_info)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print("Starting Flask-SocketIO server...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)