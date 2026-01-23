# Bug Bounty Tool - Quick Start Guide ⚡

## 30-Second Setup

```bash
cd bug_bounty_tool
python3 server.py
```

Open on your phone: `http://YOUR_IP:5000`

**That's it!** 🎉

---

## Step-by-Step (First Time)

### 1. Start the Server

```bash
cd /vercel/sandbox/bug_bounty_tool
./start.sh
```

You'll see:
```
🎯 Starting Bug Bounty Tool...

Server starting on:
  Local:   http://localhost:5000
  Network: http://192.168.1.100:5000

📱 Access from your phone:
  1. Make sure your phone is on the same WiFi network
  2. Open browser and go to: http://192.168.1.100:5000
```

### 2. Access from Your Phone

1. **Connect to WiFi**: Same network as your computer
2. **Open Browser**: Safari (iOS) or Chrome (Android)
3. **Enter URL**: Use the IP shown (e.g., `http://192.168.1.100:5000`)
4. **Bookmark**: Save for quick access

### 3. Add to Home Screen (Optional)

**iOS:**
- Tap Share → Add to Home Screen → Name it "Bug Bounty"

**Android:**
- Menu (⋮) → Add to Home Screen → Name it "Bug Bounty"

Now it launches like a native app! 📱

---

## First Use

### Add Your First Target

1. Tap **Targets** tab (bottom navigation)
2. Tap **+ Add** button
3. Fill in:
   - Name: "Example Corp"
   - URL: "https://example.com"
   - Program: "HackerOne"
   - Scope: "*.example.com"
4. Tap **Save**

### Document Your First Finding

1. Tap **Findings** tab
2. Tap **+ Add** button
3. Fill in:
   - Title: "XSS in Search"
   - Severity: Select "High"
   - Target: "example.com"
   - Description: Describe the bug
   - Steps: How to reproduce
   - Impact: What it affects
   - Bounty: Leave empty for now
4. Tap **Save**

### Take Quick Notes

1. Tap **Notes** tab
2. Tap **+ Add** button
3. Fill in:
   - Title: "Recon Notes"
   - Content: Your observations
4. Tap **Save**

---

## Daily Workflow

### Starting a Hunt

1. **Add Target** → Enter program details
2. **Take Notes** → Document reconnaissance
3. **Find Bugs** → Test the application
4. **Document Findings** → Add to Findings tab
5. **Track Progress** → Check dashboard stats

### During Testing

- **Quick Notes**: Jot down interesting endpoints
- **Screenshots**: Use phone camera for evidence
- **Findings**: Document bugs immediately
- **Updates**: Mark status as you progress

### After Submission

1. Update finding status to "submitted"
2. Add bounty amount when paid
3. Check dashboard for total earnings
4. Archive or delete old targets

---

## Common Tasks

### Backup Your Data

```bash
cd bug_bounty_tool
tar -czf backup_$(date +%Y%m%d).tar.gz data/
```

### View All Data

```bash
cat data/targets.json
cat data/findings.json
cat data/notes.json
```

### Reset Everything

```bash
rm -rf data/
# Server will create fresh files on restart
```

### Change Port

Edit `server.py`, line at bottom:
```python
run_server(port=8080)  # Change 5000 to 8080
```

---

## Troubleshooting

### Can't connect from phone?

1. **Same WiFi?** Check both devices are on same network
2. **Firewall?** Temporarily disable to test
3. **Correct IP?** Double-check the IP address
4. **Server running?** Check terminal for errors

### Server won't start?

```bash
# Check Python version (need 3.6+)
python3 --version

# Check if port is in use
lsof -i :5000

# Try different port
# Edit server.py and change port number
```

### Data not saving?

```bash
# Check permissions
ls -la data/

# Create data directory if missing
mkdir -p data
```

---

## Tips & Tricks

### 🚀 Speed Tips
- Bookmark the URL on your phone
- Add to home screen for instant access
- Keep server running in background
- Use quick notes for rapid documentation

### 📱 Mobile Tips
- Use landscape mode for forms
- Tap and hold to copy text
- Swipe to scroll long content
- Use autocomplete for repeated entries

### 🎯 Hunting Tips
- Document findings immediately
- Include all reproduction steps
- Track bounty amounts for motivation
- Regular backups of your data

### 🔒 Security Tips
- Only use on trusted networks
- Don't expose to internet
- Backup sensitive findings
- Clear data when done

---

## Next Steps

1. ✅ **Read USAGE.md** - Detailed usage guide
2. ✅ **Read FEATURES.md** - Full feature list
3. ✅ **Read README.md** - Technical details
4. ✅ **Start hunting!** - Put it to use

---

## Quick Reference

### URLs
- **Local**: http://localhost:5000
- **Network**: http://YOUR_IP:5000
- **API Docs**: See USAGE.md

### Files
- **Data**: `data/*.json`
- **Server**: `server.py`
- **Frontend**: `templates/index.html`

### Commands
```bash
# Start server
python3 server.py

# Start with script
./start.sh

# Backup data
tar -czf backup.tar.gz data/

# View stats
curl http://localhost:5000/api/stats
```

---

## Support

**Questions?** Check the documentation:
- `README.md` - Overview and installation
- `USAGE.md` - Detailed usage guide
- `FEATURES.md` - Feature descriptions

**Issues?** Check the troubleshooting section above.

**Customization?** The code is simple and well-commented!

---

**Happy Hunting!** 🎯🔍💰

*Built with ❤️ for bug bounty hunters*
