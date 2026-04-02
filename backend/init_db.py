from app import create_app
from core.database import db

app, _ = create_app()
with app.app_context():
    db.create_all()
    print("Database initialized.")
