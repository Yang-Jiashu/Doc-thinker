#!/usr/bin/env python3
"""
DocThinker UI launcher.

Usage:
  python run_ui.py
"""
import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from docthinker.ui.app import app, config


if __name__ == "__main__":
    host = getattr(config, "ui_host", "0.0.0.0")
    port = getattr(config, "ui_port", 5000)
    print()
    print("  ========================================")
    print("  DocThinker UI")
    print("  ========================================")
    print("  Chat:       http://127.0.0.1:{}/query".format(port))
    print("  KG:         http://127.0.0.1:{}/knowledge-graph".format(port))
    print("  KG alias:   http://127.0.0.1:{}/kg-viz".format(port))
    print("  ========================================")
    print()
    app.run(host=host, port=port, debug=False, use_reloader=False)
