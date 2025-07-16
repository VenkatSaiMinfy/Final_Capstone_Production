#!/usr/bin/env python3
# src/app/main.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# ────────────────────────────────────────────────────────────────
# 1) Make sure `src/` is on Python’s import path
# ────────────────────────────────────────────────────────────────
here = os.path.dirname(__file__)              # …/lead_scoring_project/src/app
project_src = os.path.abspath(os.path.join(here, ".."))  # …/lead_scoring_project/src
if project_src not in sys.path:
    sys.path.insert(0, project_src)

# ────────────────────────────────────────────────────────────────
# 2) Now do absolute imports
# ────────────────────────────────────────────────────────────────
from app import create_app

app = create_app()

if __name__ == "__main__":
    # debug=True only for local dev!
    app.run(host="0.0.0.0", port=5001, debug=True)
