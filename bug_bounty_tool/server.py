"""Bug Bounty Tool - Standalone server using only Python standard library."""

import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

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


class BugBountyHandler(BaseHTTPRequestHandler):
    """HTTP request handler for bug bounty tool."""

    def _set_headers(self, status: int = 200, content_type: str = "application/json") -> None:
        """Set response headers."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        """Handle OPTIONS requests for CORS."""
        self._set_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/targets":
            self._get_targets()
        elif path == "/api/findings":
            self._get_findings()
        elif path == "/api/notes":
            self._get_notes()
        elif path == "/api/stats":
            self._get_stats()
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def do_POST(self) -> None:
        """Handle POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()
        data = json.loads(body) if body else {}

        if self.path == "/api/targets":
            self._create_target(data)
        elif self.path == "/api/findings":
            self._create_finding(data)
        elif self.path == "/api/notes":
            self._create_note(data)
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def do_DELETE(self) -> None:
        """Handle DELETE requests."""
        if self.path.startswith("/api/targets/"):
            target_id = int(self.path.split("/")[-1])
            self._delete_target(target_id)
        elif self.path.startswith("/api/findings/"):
            finding_id = int(self.path.split("/")[-1])
            self._delete_finding(finding_id)
        elif self.path.startswith("/api/notes/"):
            note_id = int(self.path.split("/")[-1])
            self._delete_note(note_id)
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def _serve_html(self) -> None:
        """Serve the HTML file."""
        html_file = Path(__file__).parent / "templates" / "index.html"
        self._set_headers(content_type="text/html")
        with open(html_file, "rb") as f:
            self.wfile.write(f.read())

    def _get_targets(self) -> None:
        """Get all targets."""
        self._set_headers()
        targets = load_json(TARGETS_FILE)
        self.wfile.write(json.dumps(targets).encode())

    def _create_target(self, data: dict[str, Any]) -> None:
        """Create a new target."""
        targets = load_json(TARGETS_FILE)
        new_target = {
            "id": len(targets) + 1,
            "name": data.get("name", ""),
            "url": data.get("url", ""),
            "program": data.get("program", ""),
            "scope": data.get("scope", ""),
            "status": data.get("status", "active"),
            "created_at": datetime.now().isoformat(),
        }
        targets.append(new_target)
        save_json(TARGETS_FILE, targets)
        self._set_headers(201)
        self.wfile.write(json.dumps(new_target).encode())

    def _delete_target(self, target_id: int) -> None:
        """Delete a target."""
        targets = load_json(TARGETS_FILE)
        targets = [t for t in targets if t["id"] != target_id]
        save_json(TARGETS_FILE, targets)
        self._set_headers(204)

    def _get_findings(self) -> None:
        """Get all findings."""
        self._set_headers()
        findings = load_json(FINDINGS_FILE)
        self.wfile.write(json.dumps(findings).encode())

    def _create_finding(self, data: dict[str, Any]) -> None:
        """Create a new finding."""
        findings = load_json(FINDINGS_FILE)
        new_finding = {
            "id": len(findings) + 1,
            "title": data.get("title", ""),
            "severity": data.get("severity", "info"),
            "target": data.get("target", ""),
            "description": data.get("description", ""),
            "steps": data.get("steps", ""),
            "impact": data.get("impact", ""),
            "status": data.get("status", "draft"),
            "bounty": data.get("bounty", ""),
            "created_at": datetime.now().isoformat(),
        }
        findings.append(new_finding)
        save_json(FINDINGS_FILE, findings)
        self._set_headers(201)
        self.wfile.write(json.dumps(new_finding).encode())

    def _delete_finding(self, finding_id: int) -> None:
        """Delete a finding."""
        findings = load_json(FINDINGS_FILE)
        findings = [f for f in findings if f["id"] != finding_id]
        save_json(FINDINGS_FILE, findings)
        self._set_headers(204)

    def _get_notes(self) -> None:
        """Get all notes."""
        self._set_headers()
        notes = load_json(NOTES_FILE)
        self.wfile.write(json.dumps(notes).encode())

    def _create_note(self, data: dict[str, Any]) -> None:
        """Create a new note."""
        notes = load_json(NOTES_FILE)
        new_note = {
            "id": len(notes) + 1,
            "title": data.get("title", ""),
            "content": data.get("content", ""),
            "tags": data.get("tags", []),
            "created_at": datetime.now().isoformat(),
        }
        notes.append(new_note)
        save_json(NOTES_FILE, notes)
        self._set_headers(201)
        self.wfile.write(json.dumps(new_note).encode())

    def _delete_note(self, note_id: int) -> None:
        """Delete a note."""
        notes = load_json(NOTES_FILE)
        notes = [n for n in notes if n["id"] != note_id]
        save_json(NOTES_FILE, notes)
        self._set_headers(204)

    def _get_stats(self) -> None:
        """Get statistics."""
        targets = load_json(TARGETS_FILE)
        findings = load_json(FINDINGS_FILE)

        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for finding in findings:
            severity = finding.get("severity", "info").lower()
            if severity in severity_counts:
                severity_counts[severity] += 1

        total_bounty = 0.0
        for finding in findings:
            bounty = finding.get("bounty", "")
            if bounty:
                try:
                    total_bounty += float(bounty)
                except (ValueError, TypeError):
                    pass

        stats = {
            "total_targets": len(targets),
            "total_findings": len(findings),
            "severity_counts": severity_counts,
            "total_bounty": total_bounty,
        }

        self._set_headers()
        self.wfile.write(json.dumps(stats).encode())

    def log_message(self, format: str, *args: Any) -> None:
        """Override to customize logging."""
        print(f"[{self.log_date_time_string()}] {format % args}")


def run_server(port: int = 5000) -> None:
    """Run the HTTP server."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, BugBountyHandler)
    print(f"🎯 Bug Bounty Tool running on http://0.0.0.0:{port}")
    print(f"📱 Access from your phone at http://YOUR_IP:{port}")
    print("Press Ctrl+C to stop the server")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
