# 🚀 Quick Run Commands

## One-Line Command (Run Both Backend & Frontend)

```bash
(source .venv/bin/activate && cd backend && python -u app.py) & npm run dev
```

**After running:** Open `http://localhost:8080` in your browser

---

## Separate Terminal Commands (Recommended)

### Terminal 1 - Backend
```bash
source .venv/bin/activate && cd backend && python -u app.py
```

### Terminal 2 - Frontend
```bash
npm run dev
```

---

## Server URLs

| Service | URL |
|---------|-----|
| **Frontend** | `http://localhost:8080` |
| **Backend API** | `http://127.0.0.1:5001` |
| **Health Check** | `http://127.0.0.1:5001/api/health` |

---

## Stop Servers

**If using single command:**
```bash
Ctrl+C  # stops frontend
pkill -f "python app.py"  # stops backend
```

**If using separate terminals:**
```bash
Ctrl+C  # in each terminal
```

---

## Health Check

```bash
curl http://127.0.0.1:5001/api/health
```

---

## Setup (One Time)

If you need to setup the environment for the first time:

```bash
# Frontend
npm install

# Backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

If the backend still stops on missing modules, install the runtime packages into `.venv` and try again:

```bash
python -m pip install Flask-SocketIO python-socketio python-engineio psutil pynvml insightface
```

Then use the commands above to run the servers.
