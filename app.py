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
CONF_THRESHOLD = 0.80
STABILITY_FRAMES = 10
TRAINING_ARTIFACTS_DIR = 'training'

# Global variables for TFLite Interpreter
interpreter = None 
input_details = None
output_details = None
label_mapping = {}

def load_model_and_labels():
    global interpreter, label_mapping, input_details, output_details
    try:
        TFLITE_MODEL_FILENAME = os.path.join(TRAINING_ARTIFACTS_DIR, 'exercise_classifier_quant.tflite')
        LABEL_MAPPING_FILENAME = os.path.join(TRAINING_ARTIFACTS_DIR, 'label_mapping.json')
        
        if os.path.exists(TFLITE_MODEL_FILENAME):
            
            # FINAL STABLE FIX: Load interpreter directly to avoid WinError/Delegate dependency crash
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
        # FIX: Collect ALL individual fields provided by the multi-step form
        firstname = request.form.get('firstname')
        middlename = request.form.get('middlename')
        lastname = request.form.get('lastname')
        phone_num = request.form.get('phoneNum')
        gender = request.form.get('gender')
        email = request.form.get('email')
        password = request.form.get('password')
        gym_name = request.form.get('gymName')
        
        # Construct a unique identifier for the User model
        username = f"{firstname}.{lastname}.{uuid.uuid4().hex[:4]}".lower().replace(" ", "_") 

        if User.query.filter_by(email=email).first():
            flash('Email address already registered.', 'error')
            return redirect(url_for('register'))
            
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(
            firstname=firstname, middlename=middlename, lastname=lastname,
            phone_num=phone_num, gender=gender, email=email, 
            password_hash=hashed_password, gym_name=gym_name, 
            role='Gym Owner', status='active' # Assuming registration is for Gym Owner
        )
        
        db.session.add(new_user)
        db.session.commit()
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
            session['user_id'] = user.id
            session['user_role'] = user.role
            session['user_gym_name'] = user.gym_name
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
    today = datetime.utcnow().date()
    start_of_week = today - timedelta(days=today.weekday())
    start_of_month = today.replace(day=1)
    total_errors_today = db.session.query(func.count(ErrorLog.id)).filter(func.date(ErrorLog.timestamp) == today).scalar() or 0

    most_common_error_week_query = db.session.query(ErrorLog.error_type, func.count(ErrorLog.id).label('count')).filter(ErrorLog.timestamp >= start_of_week).group_by(ErrorLog.error_type).order_by(func.count(ErrorLog.id).desc()).first()
    most_common_error_week = most_common_error_week_query[0].replace('ERROR: ', '') if most_common_error_week_query else "N/A"

    total_errors_month = db.session.query(func.count(ErrorLog.id)).filter(ErrorLog.timestamp >= start_of_month).scalar() or 0

    recent_errors = db.session.query(ErrorLog, User).join(WorkoutSession, ErrorLog.session_id == WorkoutSession.id).join(User, WorkoutSession.user_id == User.id).order_by(ErrorLog.timestamp.desc()).limit(5).all()

    muscle_group_mapping = {
        'bicepCurl': 'arms', 'tricepKickback': 'arms', 'shoulderPress': 'arms',
        'lateralRaise': 'arms', 'bentOverRow': 'back',
    }
    current_month_chart_data = {'chest': 0, 'back': 0, 'legs': 0, 'arms': 0}
    errors_this_month = db.session.query(ErrorLog.exercise_name, func.count(ErrorLog.id).label('count')).filter(ErrorLog.timestamp >= start_of_month).group_by(ErrorLog.exercise_name).all()

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
# MONITOR WORKOUT (Required by house.html)
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

# --- ADMIN ROUTES ---
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    # Setup for time-based queries
    today = datetime.utcnow().date()
    start_of_week = today - timedelta(days=today.weekday())
    start_of_month = today.replace(day=1)
    
    # 1. Assigned Trainers (Kept as requested)
    current_assignments = db.session.query(Assignment).join(User).all()

    # 2. Critical Errors Today
    total_errors_today = db.session.query(func.count(ErrorLog.id)).filter(func.date(ErrorLog.timestamp) == today).scalar() or 0
    
    # 3. Common Error This Week
    most_common_error_week_query = db.session.query(ErrorLog.error_type, func.count(ErrorLog.id).label('count')).filter(ErrorLog.timestamp >= start_of_week).group_by(ErrorLog.error_type).order_by(func.count(ErrorLog.id).desc()).first()
    most_common_error_week = most_common_error_week_query[0].replace('ERROR: ', '') if most_common_error_week_query else "N/A"
    
    # 4. Overall Errors This Month
    total_errors_month = db.session.query(func.count(ErrorLog.id)).filter(ErrorLog.timestamp >= start_of_month).scalar() or 0
    
    # 5. Recent Errors Log
    recent_errors = db.session.query(ErrorLog, User).join(WorkoutSession, ErrorLog.session_id == WorkoutSession.id).join(User, WorkoutSession.user_id == User.id).order_by(ErrorLog.timestamp.desc()).limit(5).all()
    
    # 6. Muscle Group Chart Data
    muscle_group_mapping = {
        'bicepCurl': 'arms', 'tricepKickback': 'arms', 'shoulderPress': 'arms',
        'lateralRaise': 'arms', 'bentOverRow': 'back',
    }
    current_month_chart_data = {'chest': 0, 'back': 0, 'legs': 0, 'arms': 0}
    errors_this_month = db.session.query(ErrorLog.exercise_name, func.count(ErrorLog.id).label('count')).filter(ErrorLog.timestamp >= start_of_month).group_by(ErrorLog.exercise_name).all()
    
    for exercise, count in errors_this_month:
        group = muscle_group_mapping.get(exercise)
        if group and group in current_month_chart_data:
            current_month_chart_data[group] += count
            
    # CRITICAL FIX APPLIED: Pass the Python dictionary directly. 
    # Renders the correct admin template.
    return render_template(
        "admin_dashboard.html", 
        assignments=current_assignments,
        total_errors_today=total_errors_today, 
        most_common_error_week=most_common_error_week,
        total_errors_month=total_errors_month, 
        recent_errors=recent_errors,
        current_month_chart_data=current_month_chart_data # <-- FIXED: Removed json.dumps()
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

@app.route("/start_workout")
@login_required
def start_workout():
    # Placeholder for starting a workout, assumed to use a new Session model
    new_workout = WorkoutSession(user_id=session['user_id'])
    db.session.add(new_workout)
    db.session.commit()
    # Redirect to the main workout session page (which is missing in the routes provided)
    return redirect(url_for('dashboard')) 


# --- SocketIO Handlers (Partial, based on provided logic) ---
clients = {} 

@socketio.on('connect')
def handle_connect():
    # Initialize client state for both cameras
    print(f'Client connected: {request.sid}')
    # NOTE: The actual analyzer state should be robustly initialized here for two cameras
    clients[request.sid] = {
        'camera1': {'sequence_buffer': deque(maxlen=SEQUENCE_LENGTH), 'prediction_buffer': deque(maxlen=STABILITY_FRAMES), 'stable_exercise': 'neutral', 'analyzer': ExerciseAnalyzer(), 'last_form_status': None },
        'camera2': {'sequence_buffer': deque(maxlen=SEQUENCE_LENGTH), 'prediction_buffer': deque(maxlen=STABILITY_FRAMES), 'stable_exercise': 'neutral', 'analyzer': ExerciseAnalyzer(), 'last_form_status': None }
    }

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')
    clients.pop(request.sid, None)

@socketio.on('image')
def handle_image(data):
    """
    This is the lightweight handle_image, assuming process_frame_task handles the heavy lifting.
    """
    # Check for interpreter existence
    if interpreter is None: return
    try:
        sid = request.sid
        camera_id = data['camera_id']
    except (TypeError, KeyError): 
        return

    client_camera_state = clients.get(sid, {}).get(camera_id)
    if not client_camera_state:
        return 

    # --- OPTIMIZATION: Frame Skipping ---
    if client_camera_state.get('is_processing', False):
        return
        
    client_camera_state['is_processing'] = True
    
    # Start the background task (assuming a process_frame_task function is defined elsewhere)
    # NOTE: Since the full processing block was not defined in the provided code, 
    # we assume it is implemented as a background task.
    socketio.start_background_task(process_frame_task, sid, data) 

@socketio.on('start_camera')
def start_camera(data):
    sid = request.sid
    camera_id = data.get('camera_id')
    # Use a default exercise if needed, or get from data if provided
    current_exercise = data.get('current_exercise', 'neutral') 
    
    if not camera_id: 
        print(f"Error: start_camera called without camera_id for {sid}")
        return
    
    if sid in clients:
        try:
            # Create fresh instances for this camera session
            pose_instance = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            analyzer_instance = ExerciseAnalyzer(
                sequence_length=SEQUENCE_LENGTH, 
                conf_threshold=CONF_THRESHOLD, 
                stability_frames=STABILITY_FRAMES
            )
            
            # Initialize the full state for this camera
            clients[sid][camera_id] = {
                'analyzer': analyzer_instance,
                'current_exercise': current_exercise, # Store initial exercise if needed
                'is_processing': False,
                'mp_pose': pose_instance,
                # Initialize state variables used by analyzer and process_frame_task
                'sequence_buffer': deque(maxlen=SEQUENCE_LENGTH), 
                'prediction_buffer': deque(maxlen=STABILITY_FRAMES), 
                'stable_exercise': 'neutral', # Start as neutral
                'last_form_status': None
            }
            print(f"✅ Successfully started camera '{camera_id}' for client {sid}") 
        except Exception as e:
            print(f"❌ Error initializing resources for {camera_id}/{sid}: {e}") 
    else:
         print(f"Warning: Client {sid} not found during start_camera")


mp_pose = mp.solutions.pose # Ensure MediaPipe Pose is globally accessible if not already

def process_frame_task(sid, data):
    """
    This function runs in a background thread.
    It performs decoding, pose estimation, TFLite inference, and sends results back.
    """
    global interpreter, label_mapping, input_details, output_details, clients # Added 'clients'
    
    # 1. Get client/camera state
    try:
        camera_id = data['camera_id']
        client_camera_state = clients.get(sid, {}).get(camera_id)
        if not client_camera_state:
            print(f"Warning: No state found for {sid}/{camera_id}")
            return # State was torn down or never initialized
    except (KeyError, TypeError):
        print(f"Warning: Invalid data received in process_frame_task")
        return # Invalid state
        
    # 2. Decode Image
    try:
        image_data = base64.b64decode(data['image_data'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False # Performance boost
    except Exception as e:
        print(f"Error decoding image: {e}")
        if client_camera_state: client_camera_state['is_processing'] = False
        return

    # 3. Get dependencies from state
    pose = client_camera_state.get('mp_pose')
    analyzer = client_camera_state.get('analyzer')
    # Use stable_exercise from state if available, else default to 'neutral'
    current_exercise = client_camera_state.get('stable_exercise', 'neutral') 
    
    # Check if pose object exists before using it
    if not pose or not analyzer:
        print(f"Warning: Missing pose or analyzer for {sid}/{camera_id}")
        if client_camera_state: client_camera_state['is_processing'] = False
        return

    # 4. Process the frame with MediaPipe
    results = pose.process(frame_rgb)
    
    # --- FIX: Prepare landmark data for frontend ---
    landmarks_for_js = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks_for_js.append({'x': lm.x, 'y': lm.y, 'visibility': lm.visibility})
            
    # 5. Initialize emit_data (what we send back)
    emit_data = {
        'camera_id': camera_id,
        'rep_counter': analyzer.rep_counter,
        'form_status': analyzer.form_status,
        'stable_prediction': client_camera_state.get('stable_exercise', 'neutral'), # Use state value
        'landmarks': landmarks_for_js, # Include landmarks
        'debug_angles': {}
    }

    # 6. Run AI Analysis (if pose is detected and interpreter is loaded)
    if results.pose_landmarks and interpreter:
        try:
            # Pass TFLite components to the analyzer
            rep_count, form, prediction, angles = analyzer.process_frame(
                interpreter=interpreter,
                input_details=input_details,
                output_details=output_details,
                label_mapping=label_mapping,
                landmarks=results.pose_landmarks.landmark, # Pass MediaPipe landmarks directly
                current_exercise=current_exercise # Pass the current exercise state
            )
            
            # Update emit_data with results from the analyzer
            emit_data.update({
                'rep_counter': rep_count,
                'form_status': form,
                'stable_prediction': prediction, # Use the prediction from analyzer
                'debug_angles': {k: int(v) for k, v in angles.items()} 
            })
            # Update the client state with the latest stable prediction
            client_camera_state['stable_exercise'] = prediction 

        except Exception as e:
            print(f"Error during frame processing: {e}")
            emit_data['form_status'] = "Error in AI processing"

    # If no landmarks were detected, ensure analyzer status reflects this
    elif not results.pose_landmarks:
         analyzer.analyze_frame("neutral", None) # Reset analyzer state if no pose
         emit_data['form_status'] = analyzer.form_status
         emit_data['stable_prediction'] = "neutral"
         client_camera_state['stable_exercise'] = "neutral"


    # --- FIX: Ensure landmarks are always emitted ---
    # The landmarks_for_js created earlier will be empty if no pose detected, 
    # which is handled correctly by the frontend's drawSkeleton function.
    emit_data['landmarks'] = landmarks_for_js

    # 7. Send results back to the client
    socketio.emit('response', emit_data, room=sid) # Changed event name to 'response' to match monitor.html
    
    # --- Check for Audio Alert ---
    last_form = client_camera_state.get('last_form_status')
    current_form = emit_data['form_status']
    if "ERROR" in current_form and current_form != last_form:
        message = current_form.replace('ERROR: ', '') 
        socketio.emit('form_error', {'message': message, 'camera_id': camera_id}, room=sid)
        client_camera_state['last_form_status'] = current_form
    elif "ERROR" not in current_form:
        client_camera_state['last_form_status'] = None

    # --- Check for Trainer Safety Alert ---
    alert_data = analyzer.get_triggered_alert() # Assuming analyzer tracks this
    if alert_data:
        alert_data['camera_id'] = camera_id
        socketio.emit('trainer_alert', alert_data, room=sid)


    # 8. Release the lock
    if client_camera_state: client_camera_state['is_processing'] = False


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print("Starting Flask-SocketIO server...")
    # FIX: Use port 5001 to avoid socket address conflicts
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)