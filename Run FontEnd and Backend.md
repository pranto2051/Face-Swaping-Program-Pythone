# Run Frontend and Backend (Windows, macOS, Linux)

This guide explains how to install dependencies and run the project properly on each OS.

Project ports:
- Frontend (Vite): `http://localhost:8080`
- Backend (Flask): `http://127.0.0.1:5001`
- Backend health check: `http://127.0.0.1:5001/api/health`

## 1) Common Requirements (All OS)

- Node.js `18+` (recommended `20 LTS`)
- npm `9+`
- Python `3.10+` (project currently works with `3.14` in this repo)
- `pip` and `venv`
- Git

Optional but useful:
- Redis (for Celery/background queue features)

## 2) Fast Start (If You Already Installed Everything)

Run these in two terminals from project root:

Terminal 1 (Backend):
```bash
cd backend
../.venv/bin/python app.py
```

Terminal 2 (Frontend):
```bash
npm run dev
```

Then open:
- Frontend: `http://localhost:8080`
- Backend health: `http://127.0.0.1:5001/api/health`

---

## 3) Windows Setup and Run (PowerShell)

### Install tools
1. Install Node.js LTS from `https://nodejs.org`
2. Install Python from `https://python.org` (check "Add Python to PATH")
3. Install Git from `https://git-scm.com`
4. Optional Redis:
- Use WSL Redis, Docker Redis, or Redis for Windows alternatives

### One-time project setup
From project root:

```powershell
# Frontend packages
npm install

# Python virtual environment in project root
python -m venv .venv

# Activate venv
.\.venv\Scripts\Activate.ps1

# Install backend dependencies
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

If PowerShell blocks activation, run once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Run backend and frontend
Terminal 1 (Backend):
```powershell
.\.venv\Scripts\Activate.ps1
cd backend
python app.py
```

Terminal 2 (Frontend):
```powershell
npm run dev
```

### Verify
```powershell
curl http://127.0.0.1:5001/api/health
```

---

## 4) macOS Setup and Run (zsh)

### Install tools
1. Install Homebrew (if needed): `https://brew.sh`
2. Install Node + Python + Git:
```bash
brew install node python git
```
3. Optional Redis:
```bash
brew install redis
brew services start redis
```

### One-time project setup
From project root:

```bash
# Frontend packages
npm install

# Python virtual environment in project root
python3 -m venv .venv

# Activate venv
source .venv/bin/activate

# Install backend dependencies
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

### Run backend and frontend
Terminal 1 (Backend):
```bash
source .venv/bin/activate
cd backend
python app.py
```

Terminal 2 (Frontend):
```bash
npm run dev
```

### Verify
```bash
curl http://127.0.0.1:5001/api/health
```

---

## 5) Linux Setup and Run (Ubuntu/Debian style)

### Install tools
```bash
sudo apt update
sudo apt install -y nodejs npm python3 python3-venv python3-pip git
```

Optional Redis:
```bash
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

### One-time project setup
From project root:

```bash
# Frontend packages
npm install

# Python virtual environment
python3 -m venv .venv

# Activate venv
source .venv/bin/activate

# Install backend dependencies
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

### Run backend and frontend
Terminal 1 (Backend):
```bash
source .venv/bin/activate
cd backend
python app.py
```

Terminal 2 (Frontend):
```bash
npm run dev
```

### Verify
```bash
curl http://127.0.0.1:5001/api/health
```

---

## 6) Daily Run Commands (Quick Reference)

Windows:
```powershell
.\.venv\Scripts\Activate.ps1
cd backend
python app.py
```
```powershell
npm run dev
```

macOS/Linux:
```bash
source .venv/bin/activate
cd backend
python app.py
```
```bash
npm run dev
```

## 7) Common Issues and Fixes

`flask` or other module not found:
- Ensure venv is active before running backend.
- Reinstall: `python -m pip install -r backend/requirements.txt`

Port already in use (`5001` or `8080`):
- Stop existing process using that port, then rerun.

Backend not reachable from frontend:
- Check backend URL in `src/services/api.ts` is `http://127.0.0.1:5001/api`
- Check `curl http://127.0.0.1:5001/api/health`

Redis warnings or queue features not working:
- Start Redis service on your OS.

## 8) Optional: Run with Docker (Backend services)

From `backend/`:
```bash
docker compose up --build
```

This can start Redis/Postgres/backend services from `backend/docker-compose.yml`.
