# 🚀 Bug Bounty Tool - Quick Start Guide

## ✅ Server is Running!

Your bug bounty tool is **already running** and ready to use!

---

## 📱 Access from Your Phone

### Option 1: Local Access (Same Device)
Open your phone's browser and go to:
```
http://localhost:5000
```

### Option 2: Network Access (Different Device)
1. **Find your computer's IP address:**
   - **Linux/Mac:** Run `hostname -I` or `ifconfig`
   - **Windows:** Run `ipconfig`
   - Look for something like `192.168.1.x` or `10.0.0.x`

2. **Connect your phone to the same WiFi network** as your computer

3. **Open your phone's browser** and go to:
   ```
   http://YOUR_IP_ADDRESS:5000
   ```
   Example: `http://192.168.1.100:5000`

4. **Bookmark it** for quick access!

---

## 🖥️ Access from This Computer

Open your browser and go to:
```
http://localhost:5000
```

Or test it with curl:
```bash
curl http://localhost:5000
```

---

## 🎯 What You Can Do

### 1. **Manage Targets**
- Add bug bounty programs you're working on
- Track URLs, scopes, and bounty ranges
- Organize your hunting activities

### 2. **Document Findings**
- Record vulnerabilities as you discover them
- Set severity levels (Critical, High, Medium, Low, Info)
- Track bounty amounts earned
- Add detailed descriptions and reproduction steps

### 3. **Take Notes**
- Quick note-taking during reconnaissance
- Organize research by target
- Keep track of ideas and observations

### 4. **View Dashboard**
- See total targets and findings
- Track bounties earned
- View severity distribution
- Monitor your progress

---

## 🛠️ Server Management

### Check if Server is Running
```bash
curl http://localhost:5000/api/stats
```

### Stop the Server
```bash
pkill -f "python3 server.py"
```

### Start the Server
```bash
cd /vercel/sandbox/bug_bounty_tool
python3 server.py &
```

### Restart the Server
```bash
pkill -f "python3 server.py" && cd /vercel/sandbox/bug_bounty_tool && python3 server.py &
```

---

## 📂 Data Storage

All your data is stored locally in JSON files:
- **Targets:** `/vercel/sandbox/bug_bounty_tool/data/targets.json`
- **Findings:** `/vercel/sandbox/bug_bounty_tool/data/findings.json`
- **Notes:** `/vercel/sandbox/bug_bounty_tool/data/notes.json`

Your data is **private** and stays on your device. No cloud, no tracking!

---

## 🔥 Quick Test

Try adding a test target via API:
```bash
curl -X POST http://localhost:5000/api/targets \
  -H "Content-Type: application/json" \
  -d '{"name":"Test Program","url":"https://example.com","scope":"*.example.com","bounty_range":"$100-$5000"}'
```

Then view it in your browser at `http://localhost:5000`

---

## 📱 Mobile Features

The interface is optimized for phones:
- ✅ Touch-friendly buttons and forms
- ✅ Bottom navigation for easy thumb access
- ✅ Responsive design adapts to screen size
- ✅ Fast loading (< 1 second)
- ✅ Works offline after first load
- ✅ No app installation needed

---

## 🆘 Troubleshooting

### Can't Access from Phone?
1. Make sure phone and computer are on **same WiFi**
2. Check your computer's **firewall settings**
3. Verify the **IP address** is correct
4. Try using `0.0.0.0` instead of `localhost` when starting server

### Server Not Responding?
```bash
# Check if server is running
ps aux | grep "python3 server.py"

# Check if port 5000 is in use
lsof -i :5000

# Restart the server
pkill -f "python3 server.py" && cd /vercel/sandbox/bug_bounty_tool && python3 server.py &
```

### Data Not Saving?
Check file permissions:
```bash
ls -la /vercel/sandbox/bug_bounty_tool/data/
```

---

## 📚 More Documentation

- **QUICKSTART.md** - 30-second setup guide
- **USAGE.md** - Detailed usage instructions
- **FEATURES.md** - Complete feature descriptions
- **README.md** - Project overview
- **PROJECT_SUMMARY.md** - Comprehensive summary

---

## 🎯 Ready to Hunt!

Your bug bounty tool is **live and ready**. Start by:
1. Opening `http://localhost:5000` in your browser
2. Adding your first target
3. Documenting your findings
4. Tracking your bounties!

Happy hunting! 🔍💰🎯
