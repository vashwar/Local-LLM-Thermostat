#!/usr/bin/env python3
"""
LLM server lifecycle manager.
Starts llama-server on demand before evaluations, shuts down after.
"""

import logging
import subprocess
import time
import requests
import os
import signal

logger = logging.getLogger(__name__)

_process = None
_server_exe = None
_model_path = None
_port = 8080
_health_url = None


def init(server_exe: str, model_path: str, port: int = 8080):
    """Initialize with server executable path, model path, and port."""
    global _port, _health_url, _server_exe, _model_path
    _server_exe = server_exe
    _model_path = model_path
    _port = port
    _health_url = f"http://localhost:{_port}/health"
    logger.info("LLM server configured: exe=%s, model=%s, port=%d",
                _server_exe, _model_path, _port)


def is_running() -> bool:
    """Check if llama-server is responding."""
    try:
        resp = requests.get(_health_url, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def start() -> bool:
    """Start llama-server and wait for it to be ready. Returns True on success."""
    global _process

    if is_running():
        logger.info("LLM server already running")
        return True

    if not os.path.exists(_server_exe):
        logger.error("llama-server not found: %s", _server_exe)
        return False

    logger.info("Starting LLM server on port %d...", _port)

    _process = subprocess.Popen(
        [
            _server_exe,
            "-m", _model_path,
            "--port", str(_port),
            "-c", "4096",
            "-ngl", "99",
            "--log-disable"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    # Wait for server to be ready (up to 60 seconds)
    for i in range(30):
        time.sleep(2)
        if is_running():
            logger.info("LLM server ready (took %ds)", (i + 1) * 2)
            return True

    logger.error("LLM server failed to start after 60 seconds")
    stop()
    return False


def stop():
    """Stop llama-server and free GPU memory."""
    global _process

    if _process:
        logger.info("Stopping LLM server (PID %d)...", _process.pid)
        try:
            _process.terminate()
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
            _process.wait()
        _process = None
        logger.info("LLM server stopped")
    else:
        # Kill any orphaned llama-server on our port
        if is_running():
            logger.info("Killing orphaned LLM server on port %d", _port)
            try:
                os.system(f'for /f "tokens=5" %a in (\'netstat -aon ^| findstr ":{_port} " ^| findstr "LISTENING"\') do taskkill /F /PID %a >nul 2>&1')
            except Exception:
                pass
