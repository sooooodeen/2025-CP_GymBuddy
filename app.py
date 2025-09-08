from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from functools import wraps
import os

# --- App and Database Configuration ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key' 
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# --- Database Model Definition ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(100), nullable=False)
    middlename = db.Column(db.String(100), nullable=True)
    lastname = db.Column(db.String(100), nullable=False)
    phone_num = db.Column(db.String(20), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    gym_name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='active')

    assignments = db.relationship('Assignment', foreign_keys='Assignment.trainer_id', backref='trainer', lazy=True, cascade="all, delete-orphan")

class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trainer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# --- Login Required Decorators ---
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

# --- USER Routes ---
@app.route("/")
def home():
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
        role = request.form.get('role')
        gym_name = request.form.get('gymName')
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email address already registered.', 'error')
            return redirect(url_for('register'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(
            firstname=firstname, middlename=middlename, lastname=lastname,
            phone_num=phone_num, gender=gender, email=email,
            password_hash=hashed_password, role=role, gym_name=gym_name
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
            session['user_firstname'] = user.firstname
            session['user_lastname'] = user.lastname
            session['user_gym_name'] = user.gym_name
            session['user_email'] = user.email
            
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
    
@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html")

# --- ADMIN Routes ---
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    current_assignments = Assignment.query.all()
    return render_template("admin_dashboard.html", assignments=current_assignments)

@app.route("/admin/trainers")
@admin_required
def trainers():
    all_trainers = User.query.filter_by(role='Trainer').all()
    return render_template("trainers.html", trainers=all_trainers)

@app.route("/admin/monitor")
@admin_required
def admin_monitor():
    return render_template("admin_monitor.html")

@app.route("/admin/settings")
@admin_required
def admin_settings():
    return render_template("admin_settings.html")

# --- BACKEND API ROUTES FOR TRAINER ACTIONS ---
@app.route("/admin/trainers/add", methods=['POST'])
@admin_required
def add_trainer():
    data = request.get_json()
    email = data.get('email')
    
    if User.query.filter_by(email=email).first():
        return jsonify({'status': 'error', 'message': 'Email already exists.'})

    hashed_password = bcrypt.generate_password_hash(data.get('password')).decode('utf-8')
    
    new_trainer = User(
        firstname=data.get('firstname'),
        lastname=data.get('lastname'),
        email=email,
        password_hash=hashed_password,
        role='Trainer',
        phone_num='N/A',
        gender='N/A',
        gym_name=session.get('user_gym_name', 'Default Gym'),
        status='active'
    )
    db.session.add(new_trainer)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer added successfully!'})

@app.route("/admin/trainers/edit/<int:user_id>", methods=['POST'])
@admin_required
def edit_trainer(user_id):
    trainer = User.query.get_or_404(user_id)
    data = request.get_json()

    trainer.firstname = data.get('firstname')
    trainer.lastname = data.get('lastname')
    trainer.email = data.get('email')
    trainer.status = data.get('status')
    
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer updated successfully!'})

@app.route("/admin/trainers/delete/<int:user_id>", methods=['POST'])
@admin_required
def delete_trainer(user_id):
    trainer = User.query.get_or_404(user_id)
    db.session.delete(trainer)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer deleted successfully.'})

@app.route("/admin/assign/<int:trainer_id>", methods=['POST'])
@admin_required
def assign_trainer(trainer_id):
    existing_assignment = Assignment.query.filter_by(trainer_id=trainer_id).first()
    if existing_assignment:
        return jsonify({'status': 'info', 'message': 'This trainer is already assigned.'})

    assignment = Assignment(trainer_id=trainer_id)
    db.session.add(assignment)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Trainer assigned to dashboard!'})

@app.route("/admin/unassign/<int:assignment_id>", methods=['POST'])
@admin_required
def unassign_trainer(assignment_id):
    assignment = Assignment.query.get_or_404(assignment_id)
    db.session.delete(assignment)
    db.session.commit()
    flash('Trainer has been un-assigned from the dashboard.', 'success')
    return redirect(url_for('admin_dashboard'))

# --- Main Execution ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)