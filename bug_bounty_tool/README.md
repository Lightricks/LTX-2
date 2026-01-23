# Bug Bounty Tool 🎯

A mobile-friendly web application for managing bug bounty hunting activities. Track targets, document findings, and organize your research notes - all from your phone.

## Features

- **📱 Mobile-First Design**: Optimized for phone usage with touch-friendly interface
- **🎯 Target Management**: Track bug bounty programs and targets
- **🔍 Finding Documentation**: Document vulnerabilities with severity levels
- **📝 Research Notes**: Keep organized notes during your research
- **📊 Statistics Dashboard**: View your progress at a glance
- **💾 Persistent Storage**: All data saved locally in JSON files

## Quick Start

### Installation

```bash
cd bug_bounty_tool
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The application will start on `http://0.0.0.0:5000`

### Access from Your Phone

1. Make sure your phone and computer are on the same network
2. Find your computer's IP address:
   - Linux/Mac: `ifconfig` or `ip addr`
   - Windows: `ipconfig`
3. On your phone, navigate to `http://YOUR_IP:5000`

## Usage

### Targets
- Add bug bounty programs and targets you're researching
- Track URLs, program names, and scope information
- Mark targets as active or completed

### Findings
- Document discovered vulnerabilities
- Categorize by severity (Critical, High, Medium, Low, Info)
- Track bounty amounts and submission status
- Include detailed descriptions, reproduction steps, and impact

### Notes
- Quick note-taking during research
- Organize thoughts and observations
- Tag notes for easy reference

## Data Storage

All data is stored in JSON files in the `data/` directory:
- `targets.json` - Bug bounty targets
- `findings.json` - Vulnerability findings
- `notes.json` - Research notes

## Features in Detail

### Dashboard Statistics
- Total targets being tracked
- Total findings documented
- Critical severity count
- Total bounty earnings

### Mobile Optimizations
- Touch-friendly buttons and forms
- Responsive grid layout
- Bottom navigation for easy thumb access
- Smooth animations and transitions
- No horizontal scrolling

### Security Best Practices
- All data stored locally
- No external dependencies for data storage
- RESTful API design
- Input validation on forms

## Development

### Project Structure
```
bug_bounty_tool/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── data/              # JSON data storage
│   ├── targets.json
│   ├── findings.json
│   └── notes.json
└── templates/
    └── index.html     # Single-page application
```

### API Endpoints

- `GET /api/targets` - List all targets
- `POST /api/targets` - Create new target
- `DELETE /api/targets/<id>` - Delete target
- `PUT /api/targets/<id>` - Update target

- `GET /api/findings` - List all findings
- `POST /api/findings` - Create new finding
- `DELETE /api/findings/<id>` - Delete finding
- `PUT /api/findings/<id>` - Update finding

- `GET /api/notes` - List all notes
- `POST /api/notes` - Create new note
- `DELETE /api/notes/<id>` - Delete note
- `PUT /api/notes/<id>` - Update note

- `GET /api/stats` - Get statistics

## Tips for Bug Bounty Hunting

1. **Start with Reconnaissance**: Add targets and document scope carefully
2. **Document Everything**: Use notes to track your methodology
3. **Severity Matters**: Accurately categorize findings by severity
4. **Track Progress**: Use the dashboard to monitor your success
5. **Stay Organized**: Regular updates keep your workflow efficient

## License

MIT License - Feel free to modify and use for your bug bounty activities!
