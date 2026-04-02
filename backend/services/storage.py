import os
import uuid
from werkzeug.utils import secure_filename

class StorageService:
    def __init__(self, upload_folder='uploads', static_folder='static'):
        # Convert to absolute paths to avoid working directory issues
        self.upload_folder = os.path.abspath(upload_folder)
        self.static_folder = os.path.abspath(static_folder)
        os.makedirs(self.upload_folder, exist_ok=True)
        os.makedirs(self.static_folder, exist_ok=True)

    def save_upload(self, file) -> str:
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        ext = os.path.splitext(filename)[1]
        new_filename = f"{unique_id}{ext}"
        path = os.path.join(self.upload_folder, new_filename)
        file.save(path)
        return path

    def get_static_path(self, filename: str) -> str:
        return os.path.join(self.static_folder, filename)

    def get_url(self, path: str) -> str:
        # Assuming static folder is served at /static
        return f"/static/{os.path.basename(path)}"
