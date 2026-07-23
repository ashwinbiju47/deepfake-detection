# Running the Deepfake Detection Platform

This guide explains how to set up and run the project on both **macOS** and **Windows**. The project consists of a React/Vite **frontend** and a Django/Celery **backend** that depends on PostgreSQL and Redis.

---

## Prerequisites

Before starting, ensure you have the following installed on your system:
- **Node.js** (v18+ recommended)
- **Python** (v3.10+ recommended)
- **PostgreSQL** (v14+ recommended)
- **Redis** (v5+ recommended)

> **Note for Windows Users:** Redis is not officially supported on Windows natively. You can run Redis via [WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/install) or use Docker.

---

## 1. Database & Cache Setup

### PostgreSQL
1. Start your PostgreSQL server.
2. Create a database for the project:
   ```sql
   CREATE DATABASE deepfake_db;
   ```
3. Note your database credentials (username and password) to use in the backend `.env` file.

### Redis
1. Start the Redis server.
   - **macOS:** `brew services start redis` (if installed via Homebrew)
   - **Windows (WSL):** `sudo service redis-server start`
2. Redis typically runs on `localhost:6379` by default.

---

## 2. Backend Setup (Django + Celery)

Navigate to the `backend` directory in your terminal:
```bash
cd backend
```

### Create and Activate a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt / PowerShell):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure Environment Variables
Copy the example environment file and edit it:

**macOS / Linux:**
```bash
cp .env.example .env
```

**Windows:**
```cmd
copy .env.example .env
```

Open the `.env` file and update the database URL and Redis URL if necessary:
```env
DATABASE_URL=postgres://<username>:<password>@localhost:5432/deepfake_db
REDIS_URL=redis://localhost:6379/0
```

### Apply Migrations
Set up the database schema:
```bash
python manage.py migrate
```

### Run the Backend Servers
You need to run two processes for the backend (open two separate terminals, ensure the virtual environment is activated in both).

**Terminal 1 (Django ASGI Server):**
```bash
python manage.py runserver
```
*The API will be available at `http://localhost:8000`.*

**Terminal 2 (Celery Worker):**
```bash
celery -A config worker -l info --concurrency 4
```
*(On Windows, you may need to use `celery -A config worker -l info -P eventlet` or `-P solo` if you encounter issues with the default multiprocessing pool).*

---

## 3. Frontend Setup (React + Vite)

Open a new terminal and navigate to the `frontend` directory:
```bash
cd frontend
```

### Install Dependencies
```bash
npm install
```

### Run the Development Server
```bash
npm run dev
```
*The Vite server will start (typically at `http://localhost:5173`) and will automatically proxy `/api` and `/ws` requests to the Django backend running on port 8000.*

---

## Summary of Running Services

To fully run the application, you should have the following services active:
1. **PostgreSQL Service** (Running in background)
2. **Redis Service** (Running in background)
3. **Django Server** (`python manage.py runserver`) [ if backend was setup in venv then : `source .venv/bin/activate` then `python manage.py runserver` ]
4. **Celery Worker** (`celery -A config worker -l info`) [ if backend was setup in venv then : `source .venv/bin/activate` then `celery -A config worker -l info` ]
5. **Vite Frontend** (`npm run dev`)

Once everything is running, open your browser and navigate to the frontend URL (e.g., `http://localhost:5173`) to use the platform.
