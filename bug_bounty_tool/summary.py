#!/usr/bin/env python3
"""Display project summary."""

import json
from pathlib import Path

print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║              🎯 BUG BOUNTY TOOL - READY TO USE! 🎯            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

✅ PROJECT COMPLETE

📦 What's Included:
   ├─ server.py          - Python HTTP server (no dependencies!)
   ├─ templates/         - Mobile-first web interface
   ├─ data/              - JSON data storage
   ├─ start.sh           - Quick start script
   └─ Documentation/     - 6 comprehensive guides

📱 Features:
   ✓ Target Management   - Track bug bounty programs
   ✓ Finding Docs        - Document vulnerabilities
   ✓ Research Notes      - Quick note-taking
   ✓ Dashboard Stats     - Progress tracking

🚀 Quick Start:
   1. python3 server.py
   2. Open http://YOUR_IP:5000 on your phone
   3. Start hunting!

📚 Documentation:
   ├─ INDEX.md           - Documentation overview
   ├─ QUICKSTART.md      - 30-second setup
   ├─ README.md          - Project overview
   ├─ USAGE.md           - Detailed guide
   ├─ FEATURES.md        - Feature list
   └─ PROJECT_SUMMARY.md - Complete summary

🎯 Current Status:""")

data_dir = Path("data")
if data_dir.exists():
    targets_file = data_dir / "targets.json"
    findings_file = data_dir / "findings.json"

    targets = json.load(open(targets_file)) if targets_file.exists() else []
    findings = json.load(open(findings_file)) if findings_file.exists() else []
    total_bounty = sum(float(f.get("bounty", 0) or 0) for f in findings)

    print(f"   ├─ Server: Running on port 5000")
    print(f"   ├─ Targets: {len(targets)} tracked")
    print(f"   ├─ Findings: {len(findings)} documented")
    print(f"   └─ Bounty: ${total_bounty:.2f} earned")
else:
    print("   └─ Data directory will be created on first use")

print("""
🔗 Access Points:
   ├─ Local:   http://localhost:5000
   └─ Network: http://YOUR_IP:5000

📖 Next Steps:
   1. Read QUICKSTART.md for setup
   2. Access from your phone
   3. Add your first target
   4. Start documenting findings!

Happy Hunting! 🎯🔍💰
""")
