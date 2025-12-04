#!/bin/bash
# Serve documentation locally with HTTP server for hoverxref to work
cd build/html
echo "Serving documentation at http://localhost:8000"
echo "Press Ctrl+C to stop"
python -m http.server 8000
