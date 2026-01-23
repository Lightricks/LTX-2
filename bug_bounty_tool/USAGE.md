# Bug Bounty Tool - Usage Guide 📱

## Quick Start

### Start the Server

```bash
cd bug_bounty_tool
./start.sh
```

Or manually:
```bash
python3 server.py
```

### Access from Your Phone

1. **Connect to Same Network**: Ensure your phone and computer are on the same WiFi
2. **Find Your IP**: The start script will show your IP address
3. **Open Browser**: On your phone, navigate to `http://YOUR_IP:5000`
4. **Bookmark It**: Add to your home screen for quick access!

## Features Overview

### 🎯 Targets Tab
Track bug bounty programs and targets you're researching.

**What to Add:**
- **Name**: Company or application name (e.g., "Acme Corp")
- **URL**: Main target URL (e.g., "https://acme.com")
- **Program**: Bug bounty platform (e.g., "HackerOne", "Bugcrowd")
- **Scope**: In-scope domains and assets (e.g., "*.acme.com, api.acme.com")

**Use Cases:**
- Keep track of multiple programs you're working on
- Quick reference for scope information
- Organize your hunting targets

### 🔍 Findings Tab
Document vulnerabilities you discover.

**What to Add:**
- **Title**: Brief description (e.g., "XSS in Search Parameter")
- **Severity**: Critical, High, Medium, Low, or Info
- **Target**: Which target this affects
- **Description**: Detailed explanation of the vulnerability
- **Steps to Reproduce**: Clear reproduction steps
- **Impact**: What an attacker could do
- **Bounty**: Amount earned (if paid)

**Use Cases:**
- Document findings before submitting reports
- Track submission status
- Record bounty earnings
- Build your portfolio

### 📝 Notes Tab
Quick note-taking during research.

**What to Add:**
- **Title**: Note subject
- **Content**: Your observations, ideas, or findings

**Use Cases:**
- Jot down interesting endpoints
- Save reconnaissance data
- Document testing methodology
- Keep track of ideas to explore

## Mobile Tips

### Add to Home Screen (iOS)
1. Open the app in Safari
2. Tap the Share button
3. Select "Add to Home Screen"
4. Name it "Bug Bounty Tool"

### Add to Home Screen (Android)
1. Open the app in Chrome
2. Tap the menu (three dots)
3. Select "Add to Home Screen"
4. Name it "Bug Bounty Tool"

### Offline Access
- All data is stored locally on the server
- Works on local network without internet
- Data persists between sessions

## Workflow Examples

### Starting a New Program

1. **Add Target**
   - Go to Targets tab
   - Click "+ Add"
   - Fill in program details
   - Save

2. **Take Notes**
   - Switch to Notes tab
   - Document initial reconnaissance
   - Save interesting findings

3. **Document Findings**
   - When you find a bug, go to Findings tab
   - Click "+ Add"
   - Fill in all details
   - Save as draft

4. **Track Progress**
   - Check dashboard stats
   - Update finding status when submitted
   - Add bounty amount when paid

### During Active Testing

1. **Quick Notes**: Use Notes tab for rapid documentation
2. **Screenshot Reference**: Take phone screenshots of interesting behavior
3. **Finding Documentation**: Document vulnerabilities immediately
4. **Status Updates**: Mark findings as submitted/resolved

## Data Management

### Data Location
All data is stored in JSON files:
```
bug_bounty_tool/data/
├── targets.json
├── findings.json
└── notes.json
```

### Backup Your Data
```bash
# Create backup
cp -r data/ data_backup_$(date +%Y%m%d)/

# Or compress it
tar -czf bug_bounty_backup_$(date +%Y%m%d).tar.gz data/
```

### Export Data
The JSON files can be easily imported into other tools or spreadsheets.

### Reset Data
```bash
# Delete all data (be careful!)
rm -rf data/
# Server will create fresh files on next start
```

## API Reference

For advanced users who want to integrate with other tools:

### Endpoints

**Targets**
- `GET /api/targets` - List all targets
- `POST /api/targets` - Create target
- `DELETE /api/targets/{id}` - Delete target

**Findings**
- `GET /api/findings` - List all findings
- `POST /api/findings` - Create finding
- `DELETE /api/findings/{id}` - Delete finding

**Notes**
- `GET /api/notes` - List all notes
- `POST /api/notes` - Create note
- `DELETE /api/notes/{id}` - Delete note

**Statistics**
- `GET /api/stats` - Get dashboard stats

### Example API Usage

```bash
# Add a target
curl -X POST http://localhost:5000/api/targets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Example Corp",
    "url": "https://example.com",
    "program": "HackerOne",
    "scope": "*.example.com"
  }'

# Add a finding
curl -X POST http://localhost:5000/api/findings \
  -H "Content-Type: application/json" \
  -d '{
    "title": "XSS in Search",
    "severity": "high",
    "target": "example.com",
    "description": "Reflected XSS vulnerability",
    "bounty": "250"
  }'

# Get statistics
curl http://localhost:5000/api/stats
```

## Troubleshooting

### Can't Access from Phone

1. **Check Network**: Ensure both devices are on same WiFi
2. **Check Firewall**: Disable firewall temporarily to test
3. **Verify IP**: Make sure you're using the correct IP address
4. **Try Port**: Some networks block port 5000, try changing it in server.py

### Server Won't Start

1. **Check Python**: Ensure Python 3 is installed (`python3 --version`)
2. **Check Port**: Make sure port 5000 isn't already in use
3. **Check Permissions**: Ensure you have write permissions in the directory

### Data Not Saving

1. **Check Permissions**: Ensure the `data/` directory is writable
2. **Check Disk Space**: Ensure you have available disk space
3. **Check Logs**: Look for error messages in the terminal

## Security Notes

- **Local Network Only**: This tool is designed for local network use
- **No Authentication**: Anyone on your network can access it
- **Sensitive Data**: Be careful with sensitive vulnerability details
- **Backup Regularly**: Keep backups of your findings

## Best Practices

1. **Document Immediately**: Add findings as soon as you discover them
2. **Be Detailed**: Include all reproduction steps
3. **Track Everything**: Use notes for reconnaissance data
4. **Regular Backups**: Backup your data weekly
5. **Update Status**: Keep finding status current
6. **Record Bounties**: Track your earnings for motivation

## Advanced Usage

### Custom Port

Edit `server.py` and change the port:
```python
if __name__ == "__main__":
    run_server(port=8080)  # Change to your preferred port
```

### Remote Access (Use with Caution)

To access from outside your local network:
1. Set up port forwarding on your router
2. Use a VPN for secure access
3. Consider adding authentication

### Integration with Other Tools

The JSON data format makes it easy to:
- Import into spreadsheets
- Generate reports with Python scripts
- Sync with cloud storage
- Integrate with automation tools

## Support

For issues or questions:
1. Check this documentation
2. Review the README.md
3. Check the code comments in server.py
4. Test API endpoints with curl

Happy hunting! 🎯🔍
