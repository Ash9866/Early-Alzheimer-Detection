from app import create_app, db
from app.models import User

def test_database():
    # Create app instance
    app = create_app()
    
    # Use app context for database operations
    with app.app_context():
        try:
            print("Testing manual database setup...")
            
            # Test database connection
            db.engine.connect()
            print("✓ Database connection successful!")
            
            # Count users
            user_count = User.query.count()
            print(f"✓ Found {user_count} users in database")
            
            # List all users
            users = User.query.all()
            for user in users:
                print(f"  - {user.username} ({user.email}) - {user.user_type}")
            
            # Test admin login
            admin = User.query.filter_by(username='admin').first()
            if admin:
                print("✓ Admin user found!")
                # Note: We can't test password without the actual hash
            else:
                print("✗ Admin user not found!")
            
            print("\n✓ Manual database setup is working correctly!")
            
        except Exception as e:
            print(f"✗ Database test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_database()