from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    tier = db.Column(db.String(20), default='FREE') # FREE, PRO, ADMIN

class Job(db.Model):
    __tablename__ = 'jobs'
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    status = db.Column(db.String(20), default='PENDING')
    pipeline_mode = db.Column(db.String(20)) # fast, studio, cinematic
    result_url = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, server_default=db.func.now())
