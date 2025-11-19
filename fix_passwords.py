from app import create_app, db
from app.models import User

def fix_passwords():
    app = create_app()
    
    with app.app_context():
        try:
            print("Fixing user passwords...")
            
            # Update admin password
            admin = User.query.filter_by(username='admin').first()
            if admin:
                admin.set_password('admin123')
                print("✓ Admin password updated")
            
            # Update patient password
            patient = User.query.filter_by(username='john_doe').first()
            if patient:
                patient.set_password('patient123')
                print("✓ Patient password updated")
            
            db.session.commit()
            print("✓ Passwords updated successfully!")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            db.session.rollback()

if __name__ == '__main__':
    fix_passwords()