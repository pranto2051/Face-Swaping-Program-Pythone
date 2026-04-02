
## Quick Run (Project Folder Already Open)

Use these commands when your VS Code terminal is already inside the project root.

### One Command for Backend + Frontend

Copy and paste:

```bash
(source .venv/bin/activate && cd backend && python -u app.py) & npm run dev
```

This starts:
- Backend in background
- Frontend in foreground

URLs:
- Frontend: `http://localhost:8080`
- Backend API: `http://127.0.0.1:5001`
- Health: `http://127.0.0.1:5001/api/health`

### Stop Servers

```bash
Ctrl+C
pkill -f "python -u app.py"
```

---

### Separate Terminals (Optional)

Terminal 1 (Backend):

```bash
source .venv/bin/activate
cd backend
python -u app.py
```

Terminal 2 (Frontend):

```bash
npm run dev
```

### Verify Backend

```bash
curl http://127.0.0.1:5001/api/health
```