from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, IntegerField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError
from app.models import User

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already taken.')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already registered.')

class PatientForm(FlaskForm):
    full_name = StringField('Full Name', validators=[DataRequired(), Length(max=100)])
    age = IntegerField('Age', validators=[DataRequired()])
    gender = SelectField('Gender', choices=[
        ('male', 'Male'), 
        ('female', 'Female'), 
        ('other', 'Other')
    ], validators=[DataRequired()])
    contact_info = StringField('Contact Info', validators=[Length(max=200)])
    medical_history = TextAreaField('Medical History')
    submit = SubmitField('Add Patient')

class MRIUploadForm(FlaskForm):
    patient = SelectField('Patient', coerce=int, validators=[DataRequired()])
    mri_image = FileField('MRI Image', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'jpeg', 'png', 'dcm'], 'Images and DICOM files only!')
    ])
    submit = SubmitField('Upload and Analyze')