from flask import current_app  # Add this import
from flask import Blueprint, render_template, jsonify, Response, request, flash, redirect, url_for
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
import os

from app.models import db, Patient, MRIScan, Result
from app.forms import PatientForm, MRIUploadForm  # Add MRIUploadForm import

main = Blueprint('main', __name__)

@main.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Template error: {str(e)}", 500

@main.route('/test')
def test():
    return "Test route works! Flask is running."

@main.route('/debug')
def debug():
    import os
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    template_dir = os.path.join(project_root, 'templates')
    
    debug_info = {
        'project_root': project_root,
        'template_dir': template_dir,
        'template_exists': os.path.exists(template_dir),
        'templates_list': os.listdir(template_dir) if os.path.exists(template_dir) else [],
        'user_authenticated': current_user.is_authenticated
    }
    return jsonify(debug_info)

@main.route('/dashboard')
@login_required
def dashboard():
    try:
        return render_template('dashboard.html')
    except Exception as e:
        return f"Dashboard template error: {str(e)}", 500

@main.route('/patients')
@login_required
def patients():
    try:
        # Get patients based on user type
        if current_user.is_admin():
            patients_list = Patient.query.all()
        else:
            patients_list = Patient.query.filter_by(user_id=current_user.id).all()
        
        return render_template('patients.html', patients=patients_list)
    except Exception as e:
        return f"Patients template error: {str(e)}", 500

@main.route('/add_patient', methods=['GET', 'POST'])
@login_required
def add_patient():
    try:
        form = PatientForm()
        if form.validate_on_submit():
            patient = Patient(
                user_id=current_user.id,
                full_name=form.full_name.data,
                age=form.age.data,
                gender=form.gender.data,
                contact_info=form.contact_info.data,
                medical_history=form.medical_history.data
            )
            db.session.add(patient)
            db.session.commit()
            flash('Patient added successfully!', 'success')
            return redirect(url_for('main.patients'))
        
        return render_template('add_patient.html', form=form)
    except Exception as e:
        return f"Add patient template error: {str(e)}", 500
@main.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_mri():
    try:
        form = MRIUploadForm()
        
        # Populate patient choices
        if current_user.is_admin():
            form.patient.choices = [(p.id, p.full_name) for p in Patient.query.all()]
        else:
            form.patient.choices = [(p.id, p.full_name) for p in Patient.query.filter_by(user_id=current_user.id).all()]
        
        if form.validate_on_submit():
            # Save uploaded file
            file = form.mri_image.data
            filename = secure_filename(file.filename)
            unique_filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Create MRI scan record
            scan = MRIScan(
                patient_id=form.patient.data,
                user_id=current_user.id,
                image_path=unique_filename,
                original_filename=filename
            )
            db.session.add(scan)
            db.session.commit()
            
            # Perform prediction with trained model
            try:
                from app.deep_learning.predict import predict_alzheimer
                
                result = predict_alzheimer(file_path)
                
                if result['success']:
                    # ✅ FIXED: Use set_probabilities method instead of direct assignment
                    db_result = Result(
                        scan_id=scan.id,
                        prediction_class=result['predicted_class'],
                        confidence=result['confidence']
                    )
                    db_result.set_probabilities(result['probabilities'])  # Use the method
                    
                    db.session.add(db_result)
                    db.session.commit()
                    
                    flash('MRI analysis completed successfully!', 'success')
                    return redirect(url_for('main.view_result', result_id=db_result.id))
                else:
                    flash(f'Analysis failed: {result["error"]}', 'error')
                    
            except Exception as e:
                flash(f'Error during analysis: {str(e)}', 'error')
                db.session.rollback()  # Rollback on error
            
        return render_template('upload.html', form=form)
    except Exception as e:
        db.session.rollback()  # Rollback on any error
        return f"Upload template error: {str(e)}", 500

@main.route('/result/<int:result_id>')
@login_required
def view_result(result_id):
    try:
        result = Result.query.get_or_404(result_id)
        
        # ✅ FIX: Get the scan using the foreign key directly if relationship fails
        scan = MRIScan.query.get(result.scan_id)
        if not scan:
            flash('Scan not found for this result.', 'error')
            return redirect(url_for('main.dashboard'))
        
        # Check if user has permission to view this result
        if not current_user.is_admin() and scan.user_id != current_user.id:
            flash('You do not have permission to view this result.', 'error')
            return redirect(url_for('main.dashboard'))
        
        # Get patient info
        patient = Patient.query.get(scan.patient_id)
        
        return render_template('results.html', result=result, scan=scan, patient=patient)
        
    except Exception as e:
        return f"Results template error: {str(e)}", 500

@main.route('/profile')
@login_required
def profile():
    try:
        return render_template('profile.html')
    except Exception as e:
        return f"Profile template error: {str(e)}", 500

# Public routes that don't require login
@main.route('/login-test')
def login_test():
    try:
        return render_template('login.html')
    except Exception as e:
        return f"Login template error: {str(e)}", 500

@main.route('/register-test')
def register_test():
    try:
        return render_template('register.html')
    except Exception as e:
        return f"Register template error: {str(e)}", 500

# Add error handlers
@main.app_errorhandler(404)
def not_found_error(error):
    return "Route not found", 404

@main.app_errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return f"Internal server error", 500