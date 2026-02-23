#!/usr/bin/env python3
"""
NeuroAgent Clean UI v3.0
========================

A clean, minimalist interface for NeuroAgent with:
- Light theme with blue accents
- Clear file upload with progress bar
- Working chat interface
- Interactive knowledge graph

Usage:
    python launch_ui.py
    
Then open: http://localhost:5000
"""

import sys
import os
import webbrowser
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("  NeuroAgent Clean UI v3.0")
    print("=" * 60)
    
    try:
        from raganything.ui.app import app
        print("\n[1/2] Flask application loaded successfully\n")
        
        print("[2/2] Available routes:")
        routes = [
            'http://localhost:5000/',
            'http://localhost:5000/query',
            'http://localhost:5000/kg-viz',
            'http://localhost:5000/upload',
            'http://localhost:5000/config'
        ]
        for route in routes:
            print(f"      {route}")
        
        print("\n" + "=" * 60)
        print("  Server running at: http://localhost:5000")
        print("  Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        def open_browser():
            time.sleep(1.5)
            webbrowser.open('http://localhost:5000/query')
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except ImportError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)

if __name__ == '__main__':
    main()
