# 🚀 Quick Run Commands

## Single Line Command (Run Both Backend & Frontend)

```bash
(cd "/Users/md.prantoislam/Desktop/Face Swap" && source .venv/bin/activate && cd backend && python app.py) & sleep 2 && (cd "/Users/md.prantoislam/Desktop/Face Swap" && npm run dev)
```

**After running:** Open `http://localhost:8080` in your browser

---

## Separate Terminal Commands (Recommended)

### Terminal 1 - Backend
```bash
cd "/Users/md.prantoislam/Desktop/Face Swap" && source .venv/bin/activate && cd backend && python app.py
```

### Terminal 2 - Frontend
```bash
cd "/Users/md.prantoislam/Desktop/Face Swap" && npm run dev
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

Then use the commands above to run the servers.
