from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import os

# --- App and Database Configuration ---
app = Flask(__name__)
# Set a secret key for session management and flashing messages
app.secret_key = 'your_super_secret_key' 

# Configure the database
# This sets up a SQLite database file named 'database.db' in your project folder
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


# --- Database Model Definition ---
class User(db.Model):
    """Represents a user in the database."""
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def __repr__(self):
        return f'<User {self.email}>'

# --- Routes ---
@app.route("/")
def home():
    """Renders the main landing page."""
    return render_template("index.html")

@app.route("/register", methods=['GET', 'POST'])
def register():
    """Handles user registration."""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email address already registered.', 'error')
            return redirect(url_for('register'))

        # Hash the password and create a new user
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(email=email, password_hash=hashed_password)
        
        # Add the new user to the database
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template("register.html")


@app.route("/login", methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Find the user in the database
        user = User.query.filter_by(email=email).first()

        # Check if the user exists and the password is correct
        if user and bcrypt.check_password_hash(user.password_hash, password):
            # In a real app, you would set up a user session here
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password. Please try again.', 'error')
            return redirect(url_for('login'))
            
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    """Renders the dashboard page after a user successfully logs in."""
    return render_template("dashboard.html")


# --- Main Execution ---
if __name__ == "__main__":
    with app.app_context():
        # Create the database tables if they don't exist
        db.create_all()

        # --- ADDED: Create a placeholder user for testing ---
        test_user_email = "test@user.com"
        # Check if the test user already exists
        if not User.query.filter_by(email=test_user_email).first():
            print("Test user not found. Creating one...")
            # Hash the password and create the user object
            hashed_password = bcrypt.generate_password_hash("password123").decode('utf-8')
            test_user = User(email=test_user_email, password_hash=hashed_password)
            # Add to the database
            db.session.add(test_user)
            db.session.commit()
            print("Test user created successfully.")
        else:
            print("Test user already exists.")
        # --- END OF ADDED CODE ---

    app.run(debug=True)
