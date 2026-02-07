#pip install fastapi uvicorn
echo "Starting TopoMind!!!"
uvicorn topomind.server.app:app --host 0.0.0.0 --port 8000

