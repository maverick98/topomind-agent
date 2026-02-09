#!/bin/bash
echo "Starting Hawk Runtime on port 9000..."
uvicorn hawk_runtime:app --host 0.0.0.0 --port 9000 --reload
