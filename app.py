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
from dateutil.relativedelta import relativedelta # Added for monthly comparison
from werkzeug.utils import secure_filename
from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import relationship

# --- Import the shared logic ---
# CRITICAL: This line assumes you have a correct and updated analysis_logic.py file.
from analysis_logic import ExerciseAnalyzer 

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
# FIX: Use eventlet for SocketIO stability and async mode
socketio = SocketIO(app, async_mode='eventlet') 


# --- Database Model Definitions (Multi-Tenant Update) ---

class Gym(db.Model):
    """New table to store Gym information."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    # Relationships
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
    role = db.Column(db.String(50), nullable=False) # e.g., 'Gym Owner', 'Trainer'
    
    # MODIFIED: Replaced gym_name string with gym_id foreign key
    gym_id = db.Column(db.Integer, db.ForeignKey('gym.id'), nullable=False)
    
    photo_url = db.Column(db.String(200), nullable=True)
    status = db.Column(db.String(20), nullable=False, default='inactive')
    
    # Relationships
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
    
    # MODIFIED: Added gym_id to link every session to a gym
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

# --- AI MODEL AND STATE INITIALIZATION (TFLite Integration) ---
SEQUENCE_LENGTH = 90
CONF_THRESHOLD = 0.30 # Lowered threshold to fix detection
STABILITY_FRAMES = 10
TRAINING_ARTIFACTS_DIR = 'training'

interpreter = None 
input_details = None
output_details = None
label_mapping = {}
mp_pose = mp.solutions.pose # Global MediaPipe Pose instance

def load_model_and_labels():
    global interpreter, label_mapping, input_details, output_details
    try:
        TFLITE_MODEL_FILENAME = os.path.join(TRAINING_ARTIFACTS_DIR, 'exercise_classifier_quant.tflite')
        LABEL_MAPPING_FILENAME = os.path.join(TRAINING_ARTIFACTS_DIR, 'label_mapping.json')
        
        if os.path.exists(TFLITE_MODEL_FILENAME):
            interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_FILENAME)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("TFLite Interpreter loaded successfully (without Flex Delegate).")
        else:
            print(f"Error: TFLite model file not found at {TFLITE_MODEL_FILENAME}. Please run convert_model.py.")

        if os.path.exists(LABEL_MAPPING_FILENAME):
            with open(LABEL_MAPPING_FILENAME, 'r') as f:
                label_mapping = {int(k): v for k, v in json.load(f).items()}
            print("Label mapping loaded successfully.")
        else:
            print(f"Error: Label mapping file not found at {LABEL_MAPPING_FILENAME}")
            
    except Exception as e:
        print(f"Error loading model or labels: {e}") 

load_model_and_labels()


# --- Utility Functions (for Authentication and Routing) ---

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

# --- Routes (Consolidated and Fixed) ---

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

        # --- MODIFIED: Multi-Tenant Gym Logic ---
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
        new_user = User(
            firstname=firstname, middlename=middlename, lastname=lastname,
            phone_num=phone_num, gender=gender, email=email, 
            password_hash=hashed_password, 
            role='Gym Owner', status='active',
            gym_id=gym.id  # --- MODIFIED: Assign gym_id ---
        )
        
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['user_role'] = user.role
            
            # --- MODIFIED: Store gym_id and gym_name in session ---
            session['user_gym_id'] = user.gym_id
            session['user_gym_name'] = user.gym.name # Get name from relationship
            
            session['user_firstname'] = user.firstname
            session['user_lastname'] = user.lastname
            session['user_photo_url'] = user.photo_url if user.photo_url else 'src/images/Default_pfp.jpg'
            
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
def dashboard(): 
    # This is the TRAINER/USER dashboard
    # --- MODIFIED: Queries are filtered by the logged-in user's ID ---
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

    return render_template(
        "dashboard.html", 
        total_errors_today=total_errors_today,
        most_common_error_week=most_common_error_week,
        total_errors_month=total_errors_month,
        recent_errors=recent_errors,
        current_month_chart_data=current_month_chart_data 
    )

@app.route("/monitor")
@login_required
def monitor(): 
    return render_template("monitor.html")

@app.route("/errorlogpage")
@login_required
def errorlogpage(): 
    return render_template("errorlogpage.html")

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
        user.firstname = request.form.get('firstname'); user.lastname = request.form.get('lastname')
        user.email = request.form.get('email'); user.phone_num = request.form.get('phone_num')
        
        if 'gym_name' in request.form and session.get('user_role') == 'Gym Owner': 
            gym = Gym.query.get(user.gym_id)
            if gym:
                gym.name = request.form.get('gym_name')
                session['user_gym_name'] = gym.name
        
        if 'photo' in request.files:
            file = request.files['photo']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(f"{user.id}_{file.filename}")
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                user.photo_url = os.path.join('uploads/profiles', filename).replace('\\', '/')
                session['user_photo_url'] = user.photo_url # Update session photo
        db.session.commit()
        session['user_firstname'] = user.firstname; session['user_lastname'] = user.lastname
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

# --- ADMIN ROUTES (Multi-Tenant and New Features Updated) ---
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    gym_id = session['user_gym_id']
    today = datetime.utcnow().date()
    
    # --- NEW: Error Rate Comparison Logic ---
    start_of_current_month = today.replace(day=1)
    start_of_last_month = start_of_current_month - relativedelta(months=1)
    
    errors_current_month = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(
        WorkoutSession.gym_id == gym_id,
        ErrorLog.timestamp >= start_of_current_month
    ).scalar() or 0
    
    errors_last_month = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(
        WorkoutSession.gym_id == gym_id,
        ErrorLog.timestamp >= start_of_last_month,
        ErrorLog.timestamp < start_of_current_month
    ).scalar() or 0

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
    # --- END: Error Rate Logic ---

    start_of_week = today - timedelta(days=today.weekday())
    
    # --- NEW: Get the ONE assigned trainer ---
    current_assignment = db.session.query(Assignment).join(User).filter(User.gym_id == gym_id).first()
    
    total_errors_today = db.session.query(func.count(ErrorLog.id)).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, func.date(ErrorLog.timestamp) == today).scalar() or 0
    most_common_error_week_query = db.session.query(ErrorLog.error_type, func.count(ErrorLog.id).label('count')).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= start_of_week).group_by(ErrorLog.error_type).order_by(func.count(ErrorLog.id).desc()).first()
    most_common_error_week = most_common_error_week_query[0].replace('ERROR: ', '') if most_common_error_week_query else "N/A"
    total_errors_month = errors_current_month 
    recent_errors = db.session.query(ErrorLog, User).join(WorkoutSession, ErrorLog.session_id == WorkoutSession.id).join(User, WorkoutSession.user_id == User.id).filter(WorkoutSession.gym_id == gym_id).order_by(ErrorLog.timestamp.desc()).limit(5).all()

    muscle_group_mapping = {
        'bicepCurl': 'arms', 'tricepKickback': 'arms', 'shoulderPress': 'arms',
        'lateralRaise': 'arms', 'bentOverRow': 'back',
        # Add all your exercises here
    }
    current_month_chart_data = {'chest': 0, 'back': 0, 'legs': 0, 'arms': 0}
    errors_this_month = db.session.query(ErrorLog.exercise_name, func.count(ErrorLog.id).label('count')).join(WorkoutSession).filter(WorkoutSession.gym_id == gym_id, ErrorLog.timestamp >= start_of_month).group_by(ErrorLog.exercise_name).all()
    
    for exercise, count in errors_this_month:
        group = muscle_group_mapping.get(exercise)
        if group and group in current_month_chart_data:
            current_month_chart_data[group] += count
            
    return render_template(
        "admin_dashboard.html", 
        assignment=current_assignment, # Pass single assignment
        total_errors_today=total_errors_today, 
        most_common_error_week=most_common_error_week,
        total_errors_month=total_errors_month, 
        recent_errors=recent_errors,
        current_month_chart_data=current_month_chart_data,
        error_rate_change=error_rate_change,
        error_rate_color=error_rate_color,
        error_rate_status=error_rate_status
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
    
    # --- NEW: Logic to Find Last Session Time ---
    last_sessions = {}
    for trainer in all_trainers:
        last_session = WorkoutSession.query.filter_by(
            user_id=trainer.id,
            gym_id=gym_id
        ).filter(
            WorkoutSession.end_time != None 
        ).order_by(
            WorkoutSession.start_time.desc()
        ).first()
        
        if last_session:
            last_sessions[trainer.id] = last_session
    # --- END: New Logic ---

    return render_template(
        "trainers.html", 
        trainers=all_trainers, 
        assigned_trainer_ids=assigned_trainer_ids,
        last_sessions=last_sessions # Pass new data to template
    )

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
        role='Trainer', gender='N/A', 
        status='inactive',
        gym_id=session['user_gym_id'] 
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
    
    # --- NEW: Single Trainer Assignment Rule ---
    existing_assignment = db.session.query(Assignment).join(User).filter(User.gym_id == gym_id).first()
    if existing_assignment:
        return jsonify({'status': 'error', 'message': 'A trainer is already assigned. Please unassign them first.'})
    # --- END NEW RULE ---

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

# --- NEW: Trainer Session Log Routes ---
@app.route("/admin/session_log/<int:user_id>")
@admin_required
def trainer_session_log(user_id):
    gym_id = session['user_gym_id']
    trainer = User.query.filter_by(id=user_id, gym_id=gym_id).first_or_404()
    
    all_sessions = WorkoutSession.query.filter_by(
        gym_id=gym_id,
        user_id=user_id,
    ).filter(
        WorkoutSession.end_time != None # Only completed sessions
    ).order_by(WorkoutSession.start_time.desc()).all()
    
    return render_template("trainer_session_log.html", sessions=all_sessions, trainer=trainer)

@app.route("/admin/session/<int:session_id>")
@admin_required
def trainer_session_detail(session_id):
    gym_id = session['user_gym_id']
    
    session_data = db.session.query(WorkoutSession, User).join(User).filter(
        WorkoutSession.gym_id == gym_id,
        WorkoutSession.id == session_id
    ).first_or_404()
    
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
        session=session_obj,
        user=user_obj,
        duration=duration,
        normal_error_count=len(normal_errors),
        critical_error_count=len(critical_errors),
        recent_errors=errors 
    )


# --- SocketIO Handlers (Full Implementation) ---
clients = {} 

@socketio.on('connect')
def handle_connect():
    if 'user_id' not in session:
        print("Warning: Unauthenticated user tried to connect.")
        return False # Reject connection
    clients[request.sid] = {}
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in clients:
        for camera_id in list(clients[sid].keys()): 
            handle_end_session({'camera_id': camera_id}) # Gracefully end sessions
            if 'mp_pose' in clients[sid][camera_id]:
                clients[sid][camera_id]['mp_pose'].close()
    clients.pop(sid, None)
    print(f"Client disconnected: {sid}")


@socketio.on('start_camera')
def start_camera(data):
    sid = request.sid
    camera_id = data.get('camera_id')
    
    if not camera_id: return
    
    if sid in clients:
        try:
            clients[sid][camera_id] = {
                'analyzer': ExerciseAnalyzer(
                    sequence_length=SEQUENCE_LENGTH, 
                    conf_threshold=CONF_THRESHOLD, 
                    stability_frames=STABILITY_FRAMES
                ),
                'mp_pose': mp_pose.Pose(
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5
                ),
                'is_processing': False,
                'active_session_id': None, # Session not active yet
                'stable_exercise': 'neutral',
                'last_form_status': None
            }
            print(f"✅ Successfully started camera '{camera_id}' for client {sid}") 
        except Exception as e:
            print(f"❌ Error initializing resources for {camera_id}/{sid}: {e}") 

@socketio.on('stop_camera')
def stop_camera(data):
    sid = request.sid
    camera_id = data.get('camera_id')
    if not camera_id: return
    
    handle_end_session(data) # End the session for this camera
    
    if sid in clients and camera_id in clients[sid]:
        if 'mp_pose' in clients[sid][camera_id]:
            clients[sid][camera_id]['mp_pose'].close()
        clients[sid].pop(camera_id, None)
        print(f"Stopped camera '{camera_id}' for client {sid}")


# --- Session Handlers (Multi-Tenant Updated) ---
@socketio.on('start_session')
def handle_start_session(data):
    camera_id = data.get('camera_id')
    sid = request.sid
    client_camera_state = clients.get(sid, {}).get(camera_id)
    
    if 'user_id' not in session:
        print(f"Warning: Anonymous user {sid} tried to start session.")
        return

    if client_camera_state and 'analyzer' in client_camera_state:
        client_camera_state['analyzer'].reset_session()
        
        try:
            # MODIFIED: Tag new session with the user's gym_id
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
    sid = request.sid
    client_camera_state = clients.get(sid, {}).get(camera_id)
    
    if not client_camera_state or 'analyzer' not in client_camera_state:
        return

    analyzer = client_camera_state['analyzer']
    session_id = client_camera_state.get('active_session_id')

    if session_id:
        try:
            session_to_end = WorkoutSession.query.get(session_id)
            if session_to_end:
                session_to_end.end_time = datetime.utcnow()
                session_to_end.total_reps = analyzer.rep_counter
                db.session.commit()
                
                print(f"Session {session_id} ended for user {session['user_id']}.")
                emit('session_saved', {
                    'camera_id': camera_id, 
                    'reps': analyzer.rep_counter, 
                }, room=sid)
            
        except Exception as e:
            db.session.rollback()
            print(f"Error ending session {session_id} in DB: {e}")
    
    if client_camera_state:
        client_camera_state['active_session_id'] = None
    analyzer.reset_session()


# --- Main AI Processing Loop ---
def process_frame_task(sid, data):
    global interpreter, label_mapping, input_details, output_details, clients
    
    try:
        camera_id = data['camera_id']
        client_camera_state = clients.get(sid, {}).get(camera_id)
        if not client_camera_state:
            return 
    except (KeyError, TypeError):
        return 
        
    try:
        image_data = base64.b64decode(data['image_data'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False 
    except Exception as e:
        print(f"Error decoding image: {e}")
        if client_camera_state: client_camera_state['is_processing'] = False
        return

    pose = client_camera_state.get('mp_pose')
    analyzer = client_camera_state.get('analyzer')
    
    if not pose or not analyzer:
        if client_camera_state: client_camera_state['is_processing'] = False
        return

    results = pose.process(frame_rgb)
    
    landmarks_for_js = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks_for_js.append({'x': lm.x, 'y': lm.y, 'visibility': lm.visibility})
            
    emit_data = {
        'camera_id': camera_id,
        'rep_counter': analyzer.rep_counter,
        'form_status': analyzer.form_status,
        'stable_prediction': client_camera_state.get('stable_exercise', 'neutral'), 
        'landmarks': landmarks_for_js,
        'debug_angles': {}
    }

    if results.pose_landmarks and interpreter:
        try:
            rep_count, form, prediction, angles = analyzer.process_frame(
                interpreter=interpreter,
                input_details=input_details,
                output_details=output_details,
                label_mapping=label_mapping,
                landmarks=results.pose_landmarks.landmark, 
                current_exercise=client_camera_state.get('stable_exercise', 'neutral')
            )
            
            emit_data.update({
                'rep_counter': rep_count,
                'form_status': form,
                'stable_prediction': prediction, 
                'debug_angles': {k: int(v) for k, v in angles.items()} 
            })
            client_camera_state['stable_exercise'] = prediction 

            # CONTINUOUS LOGGING (Multi-Tenant)
            log_entry = analyzer.get_new_error_log()
            session_id = client_camera_state.get('active_session_id')
            
            if session_id and log_entry:
                try:
                    new_log = ErrorLog(
                        session_id=session_id, 
                        exercise_name=log_entry['exercise_name'], 
                        rep_number=log_entry['rep_number'], 
                        error_type=log_entry['error_type']
                    )
                    db.session.add(new_log)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    print(f"Error during continuous error logging: {e}")

        except Exception as e:
            print(f"Error during frame processing: {e}")
            emit_data['form_status'] = "Error in AI processing"

    elif not results.pose_landmarks:
         analyzer.analyze_frame("neutral", None)
         emit_data['form_status'] = analyzer.form_status
         emit_data['stable_prediction'] = "neutral"
         client_camera_state['stable_exercise'] = "neutral"

    emit_data['landmarks'] = landmarks_for_js

    socketio.emit('response', emit_data, room=sid)
    
    last_form = client_camera_state.get('last_form_status')
    current_form = emit_data['form_status']
    if "ERROR" in current_form and current_form != last_form:
        message = current_form.replace('ERROR: ', '') 
        socketio.emit('form_error', {'message': message, 'camera_id': camera_id}, room=sid)
        client_camera_state['last_form_status'] = current_form
    elif "ERROR" not in current_form:
        client_camera_state['last_form_status'] = None

    alert_data = analyzer.get_triggered_alert() 
    if alert_data:
        alert_data['camera_id'] = camera_id
        socketio.emit('trainer_alert', alert_data, room=sid)

    if client_camera_state: client_camera_state['is_processing'] = False


@socketio.on('image')
def handle_image(data):
    if interpreter is None: return
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
    socketio.start_background_task(process_frame_task, sid, data)


if __name__ == "__main__":
    with app.app_context():
        # Remember to delete database.db for new model changes to take effect
        db.create_all()
    print("Starting Flask-SocketIO server...")
    # FIX: Use port 5001 to avoid socket address conflicts
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)    