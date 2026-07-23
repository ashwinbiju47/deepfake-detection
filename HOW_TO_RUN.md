# How to Run the Deepfake Detection Platform

To run the platform locally, you will need to open **3 separate terminal windows**. This ensures the database, background workers, and frontend all run concurrently.

Before you begin, ensure that **PostgreSQL** and **Redis** are installed and running in the background.

---

## 🖥️ Terminal 1: The Django Backend API

This terminal runs the core REST API that serves the frontend.

### macOS & Linux
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Windows (PowerShell)
```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
*Wait until you see `Starting ASGI/Daphne version ... at http://127.0.0.1:8000/`.*

---

## ⚙️ Terminal 2: The Celery Background Worker

This terminal processes the heavy video uploads asynchronously so the website doesn't freeze. **You must activate the virtual environment in this terminal too.**

### macOS & Linux
```bash
cd backend
source .venv/bin/activate
celery -A config worker -l info --concurrency 4
```

### Windows (PowerShell)
*Note: Windows Celery requires the `solo` pool for stability.*
```powershell
cd backend
.venv\Scripts\activate
celery -A config worker -l info -P solo
```
*Wait until you see `[tasks] . detection.tasks.analyze_session` and `celery@... ready`.*

---

## 🎨 Terminal 3: The React Frontend

This terminal runs the modern user interface using Vite.

### macOS, Linux & Windows
```bash
cd frontend
npm install
npm run dev
```

---

## 🚀 Access the Platform
Once all three terminals are actively running without errors, open your web browser and navigate to:
**http://localhost:5173**

You can now upload a video, and you'll see live WebSocket logs populating in Terminal 1, and the extraction tasks executing in Terminal 2!
