
## Quick Run (After Setup) 🚀

### Single Command - Run Both Backend & Frontend

Copy and paste this single line:

```bash
(cd "/Users/md.prantoislam/Desktop/Face Swap" && source .venv/bin/activate && cd backend && python app.py) & sleep 2 && (cd "/Users/md.prantoislam/Desktop/Face Swap" && npm run dev)
```

**This will:**
- Start backend server in background
- Wait 2 seconds for backend to initialize  
- Start frontend server (you'll see the command prompt)
- Frontend will be available at: `http://localhost:8080`
- Backend API at: `http://127.0.0.1:5001`

**To stop both servers:**
```bash
Ctrl+C  # (stops frontend)
pkill -f "python app.py"  # (stops backend)
```

---

### Alternative: Run in Separate Terminals (Recommended for Development)

**Terminal 1 (Backend):**
```bash
cd "/Users/md.prantoislam/Desktop/Face Swap" && source .venv/bin/activate && cd backend && python app.py
```

**Terminal 2 (Frontend):**
```bash
cd "/Users/md.prantoislam/Desktop/Face Swap" && npm run dev
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








cd "/Users/md.prantoislam/Desktop/Face Swap" && source .venv/bin/activate && cd backend && python app.py


cd "/Users/md.prantoislam/Desktop/Face Swap" && npm run dev






Run Font End And Backend One Comand
Single Comand  command to run both backend and frontend

(cd "/Users/md.prantoislam/Desktop/Face Swap" && source .venv/bin/activate && cd backend && python app.py) & sleep 2 && (cd "/Users/md.prantoislam/Desktop/Face Swap" && npm run dev)




Or
(cd "/Users/md.prantoislam/Desktop/Face Swap" && source .venv/bin/activate && cd backend && python app.py) & (cd "/Users/md.prantoislam/Desktop/Face Swap" && npm run dev)