#!/bin/bash
echo "Starting TopoMind on port 8000..."
uvicorn topomind.server.app:app --host 0.0.0.0 --port 8000 --reload
