from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from functools import wraps
import os
from datetime import datetime
from werkzeug.utils import secure_filename

# --- App and Database Configuration ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key' 
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Configuration for file uploads ---
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads', 'profiles')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db) # Initialize Flask-Migrate

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

class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trainer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assigned_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# --- Helper function for uploads ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Decorators ---
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

# --- User Routes (Authentication) ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        gym_name = request.form.get('gymName')
        password = request.form.get('password')

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email address already registered.', 'error')
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        new_user = User(
            email=email,
            password_hash=hashed_password,
            gym_name=gym_name,
            role='Gym Owner',
            firstname=None, 
            lastname=None,
            phone_num=None,
            gender=None,
            status='active'
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
            
            if user.role == 'Gym Owner':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# --- TRAINER Routes ---
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/monitor")
@login_required
def monitor():
    return render_template("monitor.html")

@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html")


# --- Profile Routes ---
@app.route("/profile")
@login_required
def profile():
    user = User.query.get(session['user_id'])
    return render_template("profile.html", user=user)

@app.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    user = User.query.get(session['user_id'])

    # Update form data
    user.firstname = request.form.get('firstname')
    user.lastname = request.form.get('lastname')
    user.email = request.form.get('email')
    user.phone_num = request.form.get('phone_num')

    # Update gym name only if user is an admin
    if user.role == 'Gym Owner':
        user.gym_name = request.form.get('gym_name')
        session['user_gym_name'] = user.gym_name

    # Handle file upload
    if 'photo' in request.files:
        file = request.files['photo']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(f"{user.id}_{file.filename}") # Make filename unique
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            user.photo_url = os.path.join('uploads/profiles', filename).replace('\\', '/')
    
    db.session.commit()

    # Update session data so the UI reflects changes immediately
    session['user_firstname'] = user.firstname
    session['user_lastname'] = user.lastname
    
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('profile'))


@app.route('/profile/change_password', methods=['POST'])
@login_required
def change_password_submit():
    user = User.query.get(session['user_id'])
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')

    if not bcrypt.check_password_hash(user.password_hash, current_password):
        flash('Your current password was incorrect. Please try again.', 'error')
        return redirect(url_for('profile'))

    user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
    db.session.commit()

    flash('Your password has been changed successfully!', 'success')
    return redirect(url_for('profile'))


# --- ADMIN Routes ---
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    current_assignments = db.session.query(Assignment).join(User).all()
    return render_template("admin_dashboard.html", assignments=current_assignments)

@app.route("/admin/trainers")
@admin_required
def trainers():
    all_trainers = User.query.filter_by(role='Trainer').all()
    assigned_trainer_ids = [a.trainer_id for a in Assignment.query.all()]
    return render_template("trainers.html", trainers=all_trainers, assigned_trainer_ids=assigned_trainer_ids)

@app.route("/admin/monitor")
@admin_required
def admin_monitor():
    return render_template("admin_monitor.html")

# --- Trainer Management API Routes (UPDATED FOR FILE UPLOADS) ---
@app.route("/admin/trainers/add", methods=['POST'])
@admin_required
def add_trainer():
    # Switched from get_json() to request.form for multipart data
    email = request.form.get('email')
    
    if User.query.filter_by(email=email).first():
        return jsonify({'status': 'error', 'message': 'Email already exists.'})

    hashed_password = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
    
    new_trainer = User(
        firstname=request.form.get('firstname'),
        lastname=request.form.get('lastname'),
        email=email,
        phone_num=request.form.get('phone'),
        password_hash=hashed_password,
        role='Trainer',
        gender='N/A', 
        gym_name=session.get('user_gym_name', 'Default Gym'),
        status='inactive'
    )
    
    # Handle file upload for new trainer
    if 'photo' in request.files:
        file = request.files['photo']
        if file and file.filename != '' and allowed_file(file.filename):
            # We need to commit the user first to get an ID
            db.session.add(new_trainer)
            db.session.flush() # flush() assigns an ID without a full commit
            filename = secure_filename(f"{new_trainer.id}_{file.filename}")
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new_trainer.photo_url = os.path.join('uploads/profiles', filename).replace('\\', '/')

    db.session.add(new_trainer)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer added successfully!'})

@app.route("/admin/trainers/edit/<int:user_id>", methods=['POST'])
@admin_required
def edit_trainer(user_id):
    trainer = User.query.get_or_404(user_id)
    # Switched from get_json() to request.form
    trainer.firstname = request.form.get('firstname')
    trainer.lastname = request.form.get('lastname')
    trainer.email = request.form.get('email')
    trainer.phone_num = request.form.get('phone')

    # Handle file upload for editing trainer
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
    # Optional: Delete the user's photo file from the server
    if trainer.photo_url:
        try:
            os.remove(os.path.join('static', trainer.photo_url))
        except OSError as e:
            print(f"Error deleting file: {e.strerror}") # Log the error
    db.session.delete(trainer)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer deleted successfully.'})

# --- Assignment Management API Routes ---
@app.route("/admin/assign/<int:trainer_id>", methods=['POST'])
@admin_required
def assign_trainer(trainer_id):
    if Assignment.query.filter_by(trainer_id=trainer_id).first():
        return jsonify({'status': 'info', 'message': 'This trainer is already assigned.'})

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
    if trainer:
        trainer.status = 'inactive'
    
    db.session.delete(assignment)
    db.session.commit()
    flash('Trainer has been un-assigned from the dashboard.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/unassign_by_trainer/<int:trainer_id>', methods=['POST'])
@admin_required
def unassign_by_trainer_id(trainer_id):
    assignment = Assignment.query.filter_by(trainer_id=trainer_id).first()
    if assignment:
        trainer = User.query.get(trainer_id)
        if trainer:
            trainer.status = 'inactive'
        db.session.delete(assignment)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Trainer unassigned successfully.'})
    return jsonify({'status': 'error', 'message': 'Trainer was not assigned.'})

# --- Main Execution ---
if __name__ == "__main__":
    app.run(debug=True)

