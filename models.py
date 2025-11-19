from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import json

# Import db from the main package
from app import db

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    user_type = db.Column(db.String(20), nullable=False, default='patient')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    patients = db.relationship('Patient', backref='user', lazy=True)
    scans = db.relationship('MRIScan', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        return self.user_type == 'admin'
    
    def __repr__(self):
        return f'<User {self.username}>'

class Patient(db.Model):
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    contact_info = db.Column(db.String(200))
    medical_history = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    scans = db.relationship('MRIScan', backref='patient', lazy=True)
    
    def __repr__(self):
        return f'<Patient {self.full_name}>'

class MRIScan(db.Model):
    __tablename__ = 'mri_scans'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    
    # ✅ ADD THIS: Relationship with Result (one-to-one)
    result = db.relationship('Result', backref='scan', uselist=False, lazy=True)
    
    def __repr__(self):
        return f'<MRIScan {self.original_filename}>'

class Result(db.Model):
    __tablename__ = 'results'
    
    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.Integer, db.ForeignKey('mri_scans.id'), nullable=False)
    prediction_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    probabilities = db.Column(db.Text)  # Store as JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # ✅ ADD THIS: Relationship with MRIScan
    # This creates the 'scan' attribute that the template is trying to access
    
    def set_probabilities(self, prob_dict):
        """Convert dictionary to JSON string for storage"""
        self.probabilities = json.dumps(prob_dict)
    
    def get_probabilities(self):
        """Convert JSON string back to dictionary"""
        if self.probabilities:
            try:
                return json.loads(self.probabilities)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}
    
    def __repr__(self):
        return f'<Result {self.prediction_class} ({self.confidence:.2f})>'