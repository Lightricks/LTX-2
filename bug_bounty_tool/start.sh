#!/bin/bash
# Bug Bounty Tool - Quick Start Script

echo "🎯 Starting Bug Bounty Tool..."
echo ""

# Get the local IP address
IP=$(hostname -I | awk '{print $1}')

echo "Server starting on:"
echo "  Local:   http://localhost:5000"
echo "  Network: http://$IP:5000"
echo ""
echo "📱 Access from your phone:"
echo "  1. Make sure your phone is on the same WiFi network"
echo "  2. Open browser and go to: http://$IP:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 server.py
