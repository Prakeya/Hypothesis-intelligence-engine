import sys
from pathlib import Path

# Fix path to allow imports from server package
sys.path.append(str(Path(__file__).parent))

# Execute the main dashboard
app_path = Path(__file__).parent / "server" / "app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()
    exec(code, globals())
