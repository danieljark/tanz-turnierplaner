#!/usr/bin/env python3
"""
Entry point for packaged builds: starts the Flask server and opens the browser.
"""

from __future__ import annotations

import threading
import webbrowser

from app import app


def _open_browser() -> None:
    webbrowser.open("http://127.0.0.1:5000", new=2)


def main() -> None:
    threading.Timer(1.5, _open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
