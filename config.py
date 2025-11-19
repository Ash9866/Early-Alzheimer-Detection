# config.py
import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or '7904'
    
    # PostgreSQL Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://postgres:ashish7904@localhost/alzheimer_detection'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload configuration
    UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    # Model paths
    MODEL_PATH = os.path.join(basedir, 'app', 'deep_learning', 'models', 'oasis_alzheimer_model.keras')
    
    # OASIS dataset paths
    OASIS_CSV_PATH = os.path.join(basedir, 'dataset', 'oasis', 'oasis_merged.csv')
    OASIS_RAW_DATA_PATH = os.path.join(basedir, 'dataset', 'oasis', 'OAS2_RAW_part1')
    OASIS_PROCESSED_PATH = os.path.join(basedir, 'dataset', 'oasis', 'processed_images')

class DevelopmentConfig(Config):
    DEBUG = True

config = {
    'development': DevelopmentConfig,
    'default': DevelopmentConfig
}