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

# --- Database Models ---
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

# --- AI Configuration ---
SEQUENCE_LENGTH = 90
# FIXED: Lowered threshold significantly to catch exercises on wide angles
CONF_THRESHOLD = 0.15 
STABILITY_FRAMES = 5  
TRAINING_ARTIFACTS_DIR = os.path.join(basedir, 'training') 

interpreter = None 
scaler = None 
input_details = None
output_details = None
label_mapping = {}

# FIXED: Low tracking threshold to keep lock on fast movements
YOLO_CONF_THRESHOLD = 0.35 

YOLO_TO_MP = {0:0, 5:11, 6:12, 7:13, 8:14, 9:15, 10:16, 11:23, 12:24, 13:25, 14:26, 15:27, 16:28}

mp_pose = mp.solutions.pose
pose_extractor = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Adaptive Smoother Class ---
class AdaptiveSmoother:
    def __init__(self, min_alpha=0.3, max_alpha=0.9, velocity_threshold=0.02):
        # FIXED: Higher min_alpha (0.3) makes it snappier and less "laggy"
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.velocity_thresh = velocity_threshold
        self.prev_landmarks = None

    def smooth(self, current_landmarks):
        if not current_landmarks: return None
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return current_landmarks

        smoothed = []
        for i, lm in enumerate(current_landmarks):
            try:
                prev = self.prev_landmarks[i] 
                
                cx = lm['x'] if isinstance(lm, dict) else lm.x
                cy = lm['y'] if isinstance(lm, dict) else lm.y
                cz = lm['z'] if isinstance(lm, dict) else getattr(lm, 'z', 0.0)
                cv = lm['visibility'] if isinstance(lm, dict) else getattr(lm, 'visibility', 0.0)

                px, py = prev['x'], prev['y']

                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                
                alpha = self.min_alpha
                if dist > self.velocity_thresh:
                    alpha = self.max_alpha
                
                new_x = (cx * alpha) + (px * (1 - alpha))
                new_y = (cy * alpha) + (py * (1 - alpha))
                
                smoothed.append({'x': new_x, 'y': new_y, 'z': cz, 'visibility': cv})
            except Exception:
                smoothed.append(lm)
        
        self.prev_landmarks = smoothed
        return smoothed

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

        if os.path.exists(LABEL_MAPPING_FILENAME):
            with open(LABEL_MAPPING_FILENAME, 'r') as f:
                label_mapping = {int(k): v for k, v in json.load(f).items()}
            print("✅ Label mapping loaded successfully.")

        if os.path.exists(SCALER_FILENAME):
            scaler = joblib.load(SCALER_FILENAME)
            print("✅ Scaler loaded successfully.")
            
    except Exception as e:
        print(f"❌ Error loading model/scaler/labels: {e}") 

load_model_and_labels()

# --- Helper Functions ---
def generate_verification_token(email):
    return s.dumps(email, salt=app.config['SECURITY_EMAIL_SALT'])

def confirm_verification_token(token, expiration=3600):
    try:
        return s.loads(token, salt=app.config['SECURITY_EMAIL_SALT'], max_age=expiration)
    except:
        return None

def send_verification_email(user_email, token):
    verify_url = url_for('verify_account', token=token, _external=True)
    msg = Message(
        subject="Confirm Your Gym Buddy Account",
        recipients=[user_email],
        html=f"<p><a href='{verify_url}'>Verify My Email</a></p>"
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
            flash('Please log in.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('user_role') != 'Gym Owner':
            flash('Permission denied.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.context_processor
def utility_processor():
    def to_gmt8(utc_dt):
        if not utc_dt: return ""
        return utc_dt + timedelta(hours=8)
    return dict(to_gmt8=to_gmt8)

# --- Routes ---

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
            flash('Fill all fields.', 'danger')
            return redirect(url_for('register'))

        gym = Gym.query.filter_by(name=gym_name).first()
        if not gym:
            gym = Gym(name=gym_name)
            db.session.add(gym)
            db.session.commit()

        if User.query.filter_by(email=email).first():
            flash('Email taken.', 'error')
            return redirect(url_for('register'))
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(
            firstname=firstname, middlename=middlename, lastname=lastname,
            phone_num=phone_num, gender=gender, email=email, 
            password_hash=hashed_password, role='Gym Owner', 
            status='unverified', gym_id=gym.id
        )
        db.session.add(new_user)
        db.session.commit()
        
        try:
            send_verification_email(email, generate_verification_token(email))
            flash('Check email.', 'success')
        except:
            flash('Email failed.', 'warning')
        
        return redirect(url_for("verify_message")) 
    return render_template("register.html")

@app.route("/verify/<string:token>")
def verify_account(token):
    email = confirm_verification_token(token)
    if not email:
        flash('Invalid link.', 'error')
        return redirect(url_for('login'))
        
    user = User.query.filter_by(email=email).first()
    if user:
        if user.status != 'active':
            user.status = 'active'
            db.session.commit()
            flash('Verified!', 'success')
    else:
        flash('Account not found.', 'error') 
    return redirect(url_for('login'))

@app.route("/verify_message")
def verify_message():
    return render_template("verify_message.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and bcrypt.check_password_hash(user.password_hash, request.form.get('password')):
            if user.status == 'unverified':
                flash('Not verified.', 'error')
                return redirect(url_for('login'))
            
            session.clear()
            session['user_id'] = user.id
            session['user_role'] = user.role
            session['user_gym_id'] = user.gym_id
            session['user_gym_name'] = user.gym.name
            session['user_firstname'] = user.firstname
            session['user_lastname'] = user.lastname
            
            if user.role == 'Gym Owner':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials.', 'error')
    return render_template("login.html")

@app.route("/resend_verification", methods=['GET', 'POST'])
def resend_verification():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email'), status='unverified').first()
        if user:
            send_verification_email(user.email, generate_verification_token(user.email))
            flash('Sent.', 'success')
        else:
            flash('No unverified account.', 'info')
        return redirect(url_for('login'))
    return render_template("resend_form.html")

@app.route("/logout")
def logout():
    session.clear()
    flash('Logged out.', 'success')
    return redirect(url_for('login'))

@app.route("/dashboard")
@login_required
def dashboard(): 
    user_id = session['user_id']
    today = datetime.utcnow().date()
    start_of_week = today - timedelta(days=today.weekday())
    start_of_month = today.replace(day=1)
    
    total_errors_today = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.user_id == user_id, func.date(ErrorLog.timestamp) == today).scalar() or 0
    most_common = db.session.query(ErrorLog.error_type, func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.user_id == user_id, ErrorLog.timestamp >= start_of_week).group_by(ErrorLog.error_type).order_by(func.count(ErrorLog.id).desc()).first()
    most_common_error_week = most_common[0].replace('ERROR: ', '') if most_common else "N/A"
    
    total_errors_month = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.user_id == user_id, ErrorLog.timestamp >= start_of_month).scalar() or 0
    recent_errors = db.session.query(ErrorLog, User).join(WorkoutSession, ErrorLog.session_id == WorkoutSession.id).join(User, WorkoutSession.user_id == User.id).filter(User.id == user_id).order_by(ErrorLog.timestamp.desc()).limit(5).all()

    current_month_chart_data = {'chest': 0, 'back': 0, 'legs': 0, 'arms': 0}
    errors_month = db.session.query(ErrorLog.exercise_name, func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.user_id == user_id, ErrorLog.timestamp >= start_of_month).group_by(ErrorLog.exercise_name).all()
    
    mapping = {
        'bicepCurl': 'arms', 
        'lateralRaise': 'arms', 
        'shoulderPress': 'arms', 
        'dumbbellReverseFly': 'back', 
        'romanianDeadlift': 'legs'
    }
    for ex, count in errors_month:
        if mapping.get(ex) in current_month_chart_data:
            current_month_chart_data[mapping.get(ex)] += count
            
    all_sessions = WorkoutSession.query.filter_by(user_id=user_id).filter(WorkoutSession.end_time != None).order_by(WorkoutSession.start_time.desc()).all()

    return render_template("dashboard.html", total_errors_today=total_errors_today, most_common_error_week=most_common_error_week, total_errors_month=total_errors_month, recent_errors=recent_errors, current_month_chart_data=current_month_chart_data, sessions=all_sessions)

@app.route("/my_sessions")
@login_required
def my_sessions():
    if session.get('user_role') != 'Trainer':
        return redirect(url_for('admin_dashboard'))
    sessions = WorkoutSession.query.filter_by(gym_id=session['user_gym_id'], user_id=session['user_id']).filter(WorkoutSession.end_time != None).order_by(WorkoutSession.start_time.desc()).all()
    return render_template("trainer_session_log.html", sessions=sessions, trainer=db.session.get(User, session['user_id']))

@app.route("/monitor")
@login_required
def monitor():
    return render_template("monitor.html")

@app.route("/errorlogpage")
@login_required
def errorlogpage(): 
    logs = db.session.query(ErrorLog, User).join(WorkoutSession).join(User).filter(WorkoutSession.gym_id == session['user_gym_id']).order_by(ErrorLog.timestamp.desc()).all()
    js_errors = [{'id': l.id, 'userName': f"{u.firstname} {u.lastname}", 'userPhoto': url_for('static', filename=u.photo_url or 'src/images/Default_pfp.jpg'), 'errorType': l.error_type.replace('ERROR: ', ''), 'exerciseName': format_exercise_name(l.exercise_name), 'timeOfError': l.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'month': l.timestamp.strftime('%b')} for l, u in logs]
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
    user = db.session.get(User, session['user_id']) 
    if request.method == 'POST':
        user.firstname = request.form.get('firstname')
        user.lastname = request.form.get('lastname')
        user.email = request.form.get('email')
        user.phone_num = request.form.get('phone_num')
        user.gender = request.form.get('gender')
        
        if 'photo' in request.files:
            file = request.files['photo']
            if file and allowed_file(file.filename):
                fname = secure_filename(f"{user.id}_{file.filename}")
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
                user.photo_url = f"uploads/profiles/{fname}"
                session['user_photo_url'] = user.photo_url
        
        if 'gym_name' in request.form and session.get('user_role') == 'Gym Owner': 
            gym = db.session.get(Gym, user.gym_id) 
            gym.name = request.form.get('gym_name')
            session['user_gym_name'] = gym.name
        
        db.session.commit()
        session.update({'user_firstname': user.firstname, 'user_lastname': user.lastname, 'user_email': user.email, 'user_phone_num': user.phone_num})
        flash('Updated.', 'success')
        return redirect(url_for('profile'))
    return render_template("profile.html", user=user)

@app.route("/change_password", methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        data = request.get_json()
        user = db.session.get(User, session.get('user_id')) 
        if not bcrypt.check_password_hash(user.password_hash, data.get('currentPassword')):
            return jsonify({'status': 'error', 'message': 'Wrong password.'}), 400
        user.password_hash = bcrypt.generate_password_hash(data.get('newPassword')).decode('utf-8')
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Updated.'}), 200
    return render_template("change_password.html")

@app.route('/profile/change_password', methods=['POST'])
@login_required
def change_password_submit():
    user = db.session.get(User, session['user_id']) 
    if not bcrypt.check_password_hash(user.password_hash, request.form.get('current_password')):
        flash('Wrong password.', 'error')
    else:
        user.password_hash = bcrypt.generate_password_hash(request.form.get('new_password')).decode('utf-8')
        db.session.commit()
        flash('Changed.', 'success')
    return redirect(url_for('profile'))

@app.route("/delete-account", methods=['DELETE'])
@login_required
def delete_user_account():
    try:
        user_to_delete = db.session.get(User, session['user_id']) 
        db.session.delete(user_to_delete)
        db.session.commit()
        session.clear()
        return jsonify({'success': True}), 200
    except:
        db.session.rollback()
        return jsonify({'success': False}), 500

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    gym_id = session['user_gym_id']
    today = datetime.utcnow().date()
    start_of_current_month = today.replace(day=1)
    start_of_last_month = start_of_current_month - relativedelta(months=1)
    
    errors_current = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= start_of_current_month).scalar() or 0
    errors_last = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= start_of_last_month, ErrorLog.timestamp < start_of_current_month).scalar() or 0

    error_rate_change = 0
    error_rate_color = "gray"
    error_rate_status = "Month in Progress"

    if errors_last > 0:
        change_percent = ((errors_current - errors_last) / errors_last) * 100
        error_rate_color = "green" if change_percent < 0 else "red"
        error_rate_status = "Less Error" if change_percent < 0 else "More Error"
        error_rate_change = abs(round(change_percent))
    elif errors_last == 0 and errors_current > 0:
        error_rate_color = "red"
        error_rate_status = "More Error"
        error_rate_change = 100 

    start_of_week = today - timedelta(days=today.weekday())
    current_assignment = db.session.query(Assignment).join(User).filter(User.gym_id == gym_id).first()
    
    last_24_hours = datetime.utcnow() - timedelta(hours=24)
    total_errors_today = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= last_24_hours).scalar() or 0
    
    most_common = db.session.query(ErrorLog.error_type, func.count(ErrorLog.id).label('count')).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= start_of_week).group_by(ErrorLog.error_type).order_by(func.count(ErrorLog.id).desc()).first()
    most_common_error_week = most_common[0].replace('ERROR: ', '') if most_common else "N/A"
    
    recent_errors = db.session.query(ErrorLog, User).join(WorkoutSession, ErrorLog.session_id == WorkoutSession.id).join(User, WorkoutSession.user_id == User.id).filter(WorkoutSession.gym_id == gym_id).order_by(ErrorLog.timestamp.desc()).limit(5).all()
    initial_critical_errors = db.session.query(ErrorLog, User).join(WorkoutSession, ErrorLog.session_id == WorkoutSession.id).join(User, WorkoutSession.user_id == User.id).filter(WorkoutSession.gym_id == gym_id).filter(ErrorLog.error_type.contains("Repeated Error")).order_by(ErrorLog.timestamp.desc()).limit(5).all()

    current_month_chart_data = {'chest': 0, 'back': 0, 'legs': 0, 'arms': 0}
    errors_this_month = db.session.query(ErrorLog.exercise_name, func.count(ErrorLog.id).label('count')).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= start_of_current_month).group_by(ErrorLog.exercise_name).all()
    
    mapping = {
        'bicepCurl': 'arms', 
        'lateralRaise': 'arms', 
        'shoulderPress': 'arms', 
        'dumbbellReverseFly': 'back', 
        'romanianDeadlift': 'legs'
    }
    
    for ex, count in errors_this_month:
        if mapping.get(ex) in current_month_chart_data:
            current_month_chart_data[mapping.get(ex)] += count

    return render_template("admin_dashboard.html", assignment=current_assignment, total_errors_today=total_errors_today, most_common_error_week=most_common_error_week, total_errors_month=errors_current, recent_errors=recent_errors, current_month_chart_data=current_month_chart_data, error_rate_change=error_rate_change, error_rate_color=error_rate_color, error_rate_status=error_rate_status, critical_errors=initial_critical_errors)

@app.route("/admin/analytics/<int:user_id>")
@admin_required
def analytics(user_id):
    gym_id = session['user_gym_id']
    user = User.query.filter_by(id=user_id, gym_id=gym_id).first_or_404()
    sessions = WorkoutSession.query.filter_by(user_id=user.id, gym_id=gym_id).order_by(WorkoutSession.start_time.desc()).all()
    all_errors = [error.error_type for s in sessions for error in s.error_logs]
    most_common_errors = Counter(all_errors).most_common(5)
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
        gym = db.session.get(Gym, session['user_gym_id']) 
        if gym:
            gym.name = data.get('new_gym_name')
            db.session.commit()
            session['user_gym_name'] = gym.name
            return jsonify({'status': 'success', 'message': 'Updated.'})
        return jsonify({'status': 'error', 'message': 'Gym not found.'}), 404
    return render_template("admin_edit_gym_name.html")

@app.route("/admin/trainers/add", methods=['POST'])
@admin_required
def add_trainer():
    email = request.form.get('email')
    if User.query.filter_by(email=email).first():
        return jsonify({'status': 'error', 'message': 'Email exists.'})
    
    hashed_password = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
    new_trainer = User(
        firstname=request.form.get('firstname'), lastname=request.form.get('lastname'),
        email=email, phone_num=request.form.get('phone'), password_hash=hashed_password,
        role='Trainer', gender='N/A', status='inactive', gym_id=session['user_gym_id']
    )
    if 'photo' in request.files:
        file = request.files['photo']
        if file and allowed_file(file.filename):
            db.session.add(new_trainer)
            db.session.flush()
            fname = secure_filename(f"{new_trainer.id}_{file.filename}")
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
            new_trainer.photo_url = f"uploads/profiles/{fname}"
    db.session.add(new_trainer)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer added.'})

@app.route("/admin/trainers/edit/<int:user_id>", methods=['POST'])
@admin_required
def edit_trainer(user_id):
    trainer = User.query.filter_by(id=user_id, gym_id=session['user_gym_id']).first_or_404()
    trainer.firstname = request.form.get('firstname')
    trainer.lastname = request.form.get('lastname')
    trainer.email = request.form.get('email')
    trainer.phone_num = request.form.get('phone')
    if 'photo' in request.files:
        file = request.files['photo']
        if file and allowed_file(file.filename):
            fname = secure_filename(f"{trainer.id}_{file.filename}")
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
            trainer.photo_url = f"uploads/profiles/{fname}"
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Updated.'})

@app.route("/admin/trainers/delete/<int:user_id>", methods=['POST'])
@admin_required
def delete_trainer(user_id):
    trainer = User.query.filter_by(id=user_id, gym_id=session['user_gym_id']).first_or_404()
    if trainer.photo_url:
        try: os.remove(os.path.join('static', trainer.photo_url))
        except: pass
    db.session.delete(trainer)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Deleted.'})

@app.route("/admin/assign/<int:trainer_id>", methods=['POST'])
@admin_required
def assign_trainer(trainer_id):
    gym_id = session['user_gym_id']
    if db.session.query(Assignment).join(User).filter(User.gym_id == gym_id).first():
        return jsonify({'status': 'error', 'message': 'Trainer already assigned.'})
    
    trainer = User.query.filter_by(id=trainer_id, gym_id=gym_id).first()
    if not trainer: return jsonify({'status': 'error', 'message': 'Not found.'})
    
    db.session.add(Assignment(trainer_id=trainer_id))
    trainer.status = 'active'
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Assigned.'})

@app.route("/admin/unassign/<int:assignment_id>", methods=['POST'])
@admin_required
def unassign_trainer(assignment_id):
    assignment = Assignment.query.get_or_404(assignment_id)
    trainer = db.session.get(User, assignment.trainer_id) 
    if not trainer or trainer.gym_id != session['user_gym_id']:
        flash('Permission denied.', 'error')
        return redirect(url_for('admin_dashboard'))
    
    trainer.status = 'inactive'
    db.session.delete(assignment)
    db.session.commit()
    flash('Unassigned.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/unassign_by_trainer/<int:trainer_id>', methods=['POST'])
@admin_required
def unassign_by_trainer_id(trainer_id):
    assignment = Assignment.query.filter_by(trainer_id=trainer_id).first()
    if assignment:
        trainer = User.query.filter_by(id=trainer_id).first()
        trainer.status = 'inactive'
        db.session.delete(assignment)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Unassigned.'})
    return jsonify({'status': 'error', 'message': 'Not assigned.'})

@app.route("/admin/session_log/<int:user_id>")
@admin_required
def trainer_session_log(user_id):
    gym_id = session['user_gym_id']
    trainer = User.query.filter_by(id=user_id, gym_id=gym_id).first_or_404()
    sessions = WorkoutSession.query.filter_by(gym_id=gym_id, user_id=user_id).filter(WorkoutSession.end_time != None).order_by(WorkoutSession.start_time.desc()).all()
    return render_template("trainer_session_log.html", sessions=sessions, trainer=trainer)

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

    return render_template("trainer_session.html", workout_session=session_obj, user=user_obj, duration=duration, normal_error_count=len(normal_errors), critical_error_count=len(critical_errors), recent_errors=errors)

# --- Socket Handlers ---
clients = {} 

@socketio.on('connect')
def handle_connect():
    if 'user_id' not in session: return False
    gym_id = session.get('user_gym_id')
    if not gym_id: return False
    join_room(f'gym_{gym_id}')
    
    # Initialization of smoothers and ghosts is now in 'start_camera'
    clients[request.sid] = {'gym_id': gym_id}
    print(f"Client {request.sid} connected.")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in clients:
        for cam in list(clients[sid].keys()):
            if cam not in ['gym_id']: 
                handle_end_session({'camera_id': cam, 'sid_for_shutdown': sid})
        leave_room(f"gym_{clients[sid]['gym_id']}")
        del clients[sid]

@socketio.on('start_camera')
def start_camera(data):
    sid = request.sid
    cam = data.get('camera_id')
    if sid in clients and cam:
        # Initialize 'smoothers' and 'ghost_skeletons' inside the CAMERA specific dictionary
        clients[sid][cam] = {
            'analyzers': {},
            'is_processing': False,
            'active_session_id': None,
            'last_form_status': {},
            'model': YOLO('yolov8n-pose.pt'),
            'smoothers': {},  # Per-camera smoother storage
            'ghost_skeletons': {} # Per-camera ghost storage
        }

@socketio.on('stop_camera')
def stop_camera(data):
    sid = request.sid
    cam = data.get('camera_id')
    if cam:
        handle_end_session(data)
        if sid in clients and cam in clients[sid]:
            del clients[sid][cam]

@socketio.on('start_session')
def handle_start_session(data):
    sid = request.sid
    cam = data.get('camera_id')
    state = clients.get(sid, {}).get(cam)
    if state:
        for a in state['analyzers'].values():
            a.reset_session()
        try:
            ns = WorkoutSession(user_id=session['user_id'], gym_id=session['user_gym_id'])
            db.session.add(ns)
            db.session.commit()
            state['active_session_id'] = ns.id
            emit('session_started', {'camera_id': cam}, room=sid)
        except:
            db.session.rollback()

@socketio.on('end_session')
def handle_end_session(data):
    sid = data.get('sid_for_shutdown', request.sid)
    cam = data.get('camera_id')
    state = clients.get(sid, {}).get(cam)
    if not state: return
    
    sess_id = state.get('active_session_id')
    if sess_id:
        try:
            with app.app_context():
                s = db.session.get(WorkoutSession, sess_id) 
                if s:
                    s.end_time = datetime.utcnow()
                    reps = sum(a.rep_counter for a in state['analyzers'].values())
                    s.total_reps = reps
                    db.session.commit()
                    if not data.get('sid_for_shutdown'):
                        emit('session_saved', {'camera_id': cam, 'reps': reps}, room=sid)
        except:
            db.session.rollback()
    
    state['active_session_id'] = None
    for a in state['analyzers'].values():
        a.reset_session()

def process_frame_task(sid, data, session_context):
    gym_id = session_context.get('gym_id')
    try:
        cam = data['camera_id']
        state = clients.get(sid, {}).get(cam)
    except: return
    if not state: return

    try:
        # OPTIMIZATION: Use OpenCV instead of PIL for faster decoding
        img_data = base64.b64decode(data['image_data'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Calculate Aspect Ratio Dimensions for AI Normalization
        h, w, _ = frame.shape
        max_dim = max(h, w)
        
        # YOLO expects RGB, OpenCV gives BGR. Convert it.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
    except Exception as e:
        print(f"Frame Error: {e}")
        state['is_processing'] = False
        return

    # Use Independent Model
    model = state.get('model')
    if not model: 
        state['is_processing'] = False
        return

    try:
        # ADDED imgsz=640 to boost accuracy
        results = model.track(frame_rgb, verbose=False, conf=YOLO_CONF_THRESHOLD, persist=True, tracker="bytetrack.yaml", imgsz=640)
    except:
        state['is_processing'] = False
        return

    current_data = []
    active_ids = []
    
    if results[0].boxes and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        kpts = results[0].keypoints
        active_ids = track_ids

        for i, tid in enumerate(track_ids):
            if tid not in state['analyzers']:
                state['analyzers'][tid] = ExerciseAnalyzer(SEQUENCE_LENGTH, CONF_THRESHOLD, STABILITY_FRAMES)
                state['last_form_status'][tid] = None
            
            if tid not in state['smoothers']:
                state['smoothers'][tid] = AdaptiveSmoother(0.3, 0.9, 0.02) # Snappier

            analyzer = state['analyzers'][tid]
            smoother = state['smoothers'][tid]
            
            lms_ui = [{'x': 0.0, 'y': 0.0, 'z': 0.0, 'visibility': 0.0} for _ in range(33)]
            lms_ai = [{'x': 0.0, 'y': 0.0, 'z': 0.0, 'visibility': 0.0} for _ in range(33)]

            if kpts and kpts.conf is not None:
                xy = kpts.xy[i].cpu().numpy()
                conf = kpts.conf[i].cpu().numpy()
                for yidx, midx in YOLO_TO_MP.items():
                    if yidx < len(conf) and conf[yidx] > 0.5:
                        raw_x, raw_y = xy[yidx][0], xy[yidx][1]
                        
                        # Standard 0-1 Normalization for UI Drawing
                        lms_ui[midx] = {'x': raw_x/w, 'y': raw_y/h, 'z': 0.0, 'visibility': float(conf[yidx])}
                        
                        # "Magic" Aspect Ratio Corrected Normalization for AI
                        # This centers the skeleton in a square virtual frame
                        norm_x_ai = (raw_x + (max_dim - w) / 2) / max_dim
                        norm_y_ai = (raw_y + (max_dim - h) / 2) / max_dim
                        lms_ai[midx] = {'x': norm_x_ai, 'y': norm_y_ai, 'z': 0.0, 'visibility': float(conf[yidx])}

            smoothed_lms = smoother.smooth(lms_ui)
            smoothed_lms_ai = smoother.smooth(lms_ai) # Smooth AI inputs too
            
            # Update Ghost Memory
            state['ghost_skeletons'][tid] = {'lms': smoothed_lms, 'ttl': 5, 'an': analyzer} 

            p_resp = {
                'track_id': tid,
                'rep_counter': analyzer.rep_counter,
                'form_status': analyzer.form_status,
                'stable_prediction': analyzer.stable_prediction,
                'landmarks': smoothed_lms,
                'debug_angles': {}
            }

            if interpreter and scaler:
                try:
                    # PASS THE CORRECTED LANDMARKS TO THE ANALYZER
                    reps, form, pred, ang = analyzer.process_frame(interpreter, input_details, output_details, label_mapping, smoothed_lms_ai, analyzer.stable_prediction, scaler)
                    p_resp.update({'rep_counter': reps, 'form_status': form, 'stable_prediction': pred, 'debug_angles': {k: int(v) for k, v in ang.items()}})
                    
                    last_form = state['last_form_status'].get(tid)
                    is_new = "ERROR" in form and form != last_form
                    log = analyzer.get_new_error_log()
                    sess_id = state.get('active_session_id')

                    if (log or is_new) and not sess_id:
                        with app.app_context():
                            ns = WorkoutSession(user_id=session_context['user_id'], gym_id=gym_id)
                            db.session.add(ns)
                            db.session.commit()
                            state['active_session_id'] = ns.id
                    
                    if sess_id:
                        dat = None
                        if log: dat = {'e': log['exercise_name'], 'r': log['rep_number'], 't': log['error_type']}
                        elif is_new: dat = {'e': pred, 'r': reps, 't': form.replace('ERROR: ', '')}
                        if dat:
                            with app.app_context():
                                db.session.add(ErrorLog(session_id=sess_id, exercise_name=dat['e'], rep_number=dat['r'], error_type=dat['t']))
                                db.session.commit()
                    
                    if is_new:
                        with app.app_context():
                            socketio.emit('form_error', {'message': f"Person {tid}: {form.replace('ERROR: ', '')}", 'camera_id': cam}, room=f'gym_{gym_id}')
                        state['last_form_status'][tid] = form
                    elif "ERROR" not in form:
                        state['last_form_status'][tid] = None

                    alert = analyzer.get_triggered_alert()
                    if alert: 
                        with app.app_context():
                            socketio.emit('trainer_alert', alert, room=f'gym_{gym_id}')
                except: pass
            current_data.append(p_resp)

    # Ghost Handling
    for gid, ghost in list(state['ghost_skeletons'].items()):
        if gid not in active_ids:
            if ghost['ttl'] > 0:
                ghost['ttl'] -= 1
                current_data.append({
                    'track_id': gid,
                    'rep_counter': ghost['an'].rep_counter,
                    'form_status': ghost['an'].form_status,
                    'stable_prediction': ghost['an'].stable_prediction,
                    'landmarks': ghost['lms'],
                    'debug_angles': {}
                })
            else:
                del state['ghost_skeletons'][gid]

    # FIXED: Use socketio.emit instead of just emit
    socketio.emit('response', {'camera_id': cam, 'people': current_data}, room=sid)
    state['is_processing'] = False

@socketio.on('image')
def handle_image(data):
    try:
        sid = request.sid
        cam = data['camera_id']
    except: return
    state = clients.get(sid, {}).get(cam)
    if not state or state.get('is_processing', False): return
    state['is_processing'] = True
    ctx = {'user_id': session.get('user_id'), 'gym_id': session.get('user_gym_id'), 'firstname': session.get('user_firstname'), 'lastname': session.get('user_lastname')}
    socketio.start_background_task(process_frame_task, sid, data, ctx)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print("Starting Flask-SocketIO server...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)