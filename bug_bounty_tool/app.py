"""Bug Bounty Tool - Mobile-friendly web application for bug bounty hunting."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Data directory
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TARGETS_FILE = DATA_DIR / "targets.json"
FINDINGS_FILE = DATA_DIR / "findings.json"
NOTES_FILE = DATA_DIR / "notes.json"


def load_json(filepath: Path) -> list[dict[str, Any]]:
    """Load JSON data from file."""
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return []


def save_json(filepath: Path, data: list[dict[str, Any]]) -> None:
    """Save JSON data to file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


@app.route("/")
def index() -> str:
    """Render main page."""
    return render_template("index.html")


@app.route("/api/targets", methods=["GET", "POST"])
def targets() -> Any:
    """Handle targets endpoint."""
    if request.method == "GET":
        return jsonify(load_json(TARGETS_FILE))

    data = request.json
    targets_data = load_json(TARGETS_FILE)
    new_target = {
        "id": len(targets_data) + 1,
        "name": data["name"],
        "url": data["url"],
        "program": data.get("program", ""),
        "scope": data.get("scope", ""),
        "status": data.get("status", "active"),
        "created_at": datetime.now().isoformat(),
    }
    targets_data.append(new_target)
    save_json(TARGETS_FILE, targets_data)
    return jsonify(new_target), 201


@app.route("/api/targets/<int:target_id>", methods=["DELETE", "PUT"])
def target_detail(target_id: int) -> Any:
    """Handle individual target operations."""
    targets_data = load_json(TARGETS_FILE)

    if request.method == "DELETE":
        targets_data = [t for t in targets_data if t["id"] != target_id]
        save_json(TARGETS_FILE, targets_data)
        return "", 204

    if request.method == "PUT":
        data = request.json
        for target in targets_data:
            if target["id"] == target_id:
                target.update(data)
                save_json(TARGETS_FILE, targets_data)
                return jsonify(target)
        return jsonify({"error": "Target not found"}), 404

    return jsonify({"error": "Method not allowed"}), 405


@app.route("/api/findings", methods=["GET", "POST"])
def findings() -> Any:
    """Handle findings endpoint."""
    if request.method == "GET":
        return jsonify(load_json(FINDINGS_FILE))

    data = request.json
    findings_data = load_json(FINDINGS_FILE)
    new_finding = {
        "id": len(findings_data) + 1,
        "title": data["title"],
        "severity": data["severity"],
        "target": data.get("target", ""),
        "description": data.get("description", ""),
        "steps": data.get("steps", ""),
        "impact": data.get("impact", ""),
        "status": data.get("status", "draft"),
        "bounty": data.get("bounty", ""),
        "created_at": datetime.now().isoformat(),
    }
    findings_data.append(new_finding)
    save_json(FINDINGS_FILE, findings_data)
    return jsonify(new_finding), 201


@app.route("/api/findings/<int:finding_id>", methods=["DELETE", "PUT"])
def finding_detail(finding_id: int) -> Any:
    """Handle individual finding operations."""
    findings_data = load_json(FINDINGS_FILE)

    if request.method == "DELETE":
        findings_data = [f for f in findings_data if f["id"] != finding_id]
        save_json(FINDINGS_FILE, findings_data)
        return "", 204

    if request.method == "PUT":
        data = request.json
        for finding in findings_data:
            if finding["id"] == finding_id:
                finding.update(data)
                save_json(FINDINGS_FILE, findings_data)
                return jsonify(finding)
        return jsonify({"error": "Finding not found"}), 404

    return jsonify({"error": "Method not allowed"}), 405


@app.route("/api/notes", methods=["GET", "POST"])
def notes() -> Any:
    """Handle notes endpoint."""
    if request.method == "GET":
        return jsonify(load_json(NOTES_FILE))

    data = request.json
    notes_data = load_json(NOTES_FILE)
    new_note = {
        "id": len(notes_data) + 1,
        "title": data["title"],
        "content": data["content"],
        "tags": data.get("tags", []),
        "created_at": datetime.now().isoformat(),
    }
    notes_data.append(new_note)
    save_json(NOTES_FILE, notes_data)
    return jsonify(new_note), 201


@app.route("/api/notes/<int:note_id>", methods=["DELETE", "PUT"])
def note_detail(note_id: int) -> Any:
    """Handle individual note operations."""
    notes_data = load_json(NOTES_FILE)

    if request.method == "DELETE":
        notes_data = [n for n in notes_data if n["id"] != note_id]
        save_json(NOTES_FILE, notes_data)
        return "", 204

    if request.method == "PUT":
        data = request.json
        for note in notes_data:
            if note["id"] == note_id:
                note.update(data)
                save_json(NOTES_FILE, notes_data)
                return jsonify(note)
        return jsonify({"error": "Note not found"}), 404

    return jsonify({"error": "Method not allowed"}), 405


@app.route("/api/stats", methods=["GET"])
def stats() -> Any:
    """Get statistics."""
    targets_data = load_json(TARGETS_FILE)
    findings_data = load_json(FINDINGS_FILE)

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    for finding in findings_data:
        severity = finding.get("severity", "info").lower()
        if severity in severity_counts:
            severity_counts[severity] += 1

    total_bounty = sum(
        float(f.get("bounty", 0) or 0) for f in findings_data if f.get("bounty")
    )

    return jsonify(
        {
            "total_targets": len(targets_data),
            "total_findings": len(findings_data),
            "severity_counts": severity_counts,
            "total_bounty": total_bounty,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
