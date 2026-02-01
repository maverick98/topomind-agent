#!/bin/bash

echo "Setting PYTHONPATH to project root..."
export PYTHONPATH="$(pwd)"

echo "Running TopoMind agent..."
python examples/run_agent.py
