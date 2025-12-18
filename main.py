import argparse
import os
import uvicorn
import requests
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import your existing Green Agent
from src.green_agent.green_agent import GreenAgent

# --- 1. REMOTE ADAPTER ---
class RemoteAgentAdapter:
    """
    Connects the Green Agent to a remote White Agent (A2A over HTTP).
    """
    def __init__(self, target_url):
        self.target_url = target_url.rstrip('/')
        if not self.target_url.endswith("/agent/tasks") and not self.target_url.endswith("/tasks"):
             # Basic heuristic to ensure endpoint validity
             pass 

    def receive_task(self, message, image_path=None):
        payload = {
            "message": message,
            "image_path": image_path if image_path else None 
        }
        try:
            # We send the path to the white agent. 
            # Since we are running locally, the absolute path works for both processes.
            response = requests.post(self.target_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Remote Connection Failed: {str(e)}"}

# --- 2. FASTAPI SERVER ---
app = FastAPI()

# MOUNT STATIC FILES
# 1. Output: So you can see the HTML report in the browser
# 2. Dataset: So the HTML report can load the images
output_dir = os.path.join(os.getcwd(), "output")
dataset_dir = os.path.join(os.getcwd(), "dataset")

os.makedirs(output_dir, exist_ok=True)

app.mount("/report", StaticFiles(directory=output_dir), name="report")
app.mount("/dataset", StaticFiles(directory=dataset_dir), name="dataset")

# Initialize Green Agent ONCE
green_agent = GreenAgent(model_name="llama3.2")

@app.post("/assess")
async def assess_remote(request: Request):
    """
    Endpoint to trigger an assessment run against a specific URL.
    """
    data = await request.json()
    target_url = data.get("target_url")
    
    # --- FIX: DYNAMIC AGENT NAMING ---
    # 1. Check if name is explicitly provided
    agent_name = data.get("agent_name")
    
    # 2. If not, try to extract from URL (e.g. .../agent/moondream/tasks)
    if not agent_name and target_url:
        try:
            if "/agent/" in target_url:
                # Split by '/agent/' and take the next segment
                parts = target_url.split("/agent/")
                if len(parts) > 1:
                    agent_name = parts[1].split("/")[0]
        except:
            pass
    
    # 3. Fallback
    if not agent_name:
        agent_name = "Remote_Candidate"

    if not target_url:
        return {"status": "error", "result": "Missing 'target_url' in request."}

    # Connect the adapter
    adapter = RemoteAgentAdapter(target_url)
    green_agent.connect_white_agent(adapter)
    
    # Use Absolute Path for Dataset
    dataset_path = os.path.abspath(os.path.join(os.getcwd(), "dataset"))
    
    if not os.path.exists(dataset_path):
        return {"status": "error", "result": "Dataset folder not found."}

    try:
        # Run Assessment
        # We pass the dynamic agent_name here so the HTML report is correct
        print(f"🚦 Starting Assessment for {agent_name}...")
        result = green_agent.run_assessment(dataset_path, limit=4, agent_name=agent_name)
        
        # Generate HTML Artifacts
        green_agent.generate_artifacts(output_dir)
        
        # Build the Clickable Link
        base_url = str(request.base_url).rstrip("/")
        report_link = f"{base_url}/report/leaderboard.html"
        
        print(f"✅ Assessment Complete! Score: {result.get('overall_score_percent')}%")
        
        return {
            "status": "success",
            "agent": agent_name,
            "verdict": result.get('overall_grade'),
            "score": result.get('overall_score_percent'),
            "analysis": result.get('analysis'),
            "view_report_here": report_link
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "result": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Green Agent Controller")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    args, unknown = parser.parse_known_args()

    # Priority:
    # 1. Command Line Flag (--port 1234)
    # 2. AgentBeats Variable ($AGENT_PORT)
    # 3. Standard Cloud Variable ($PORT)
    # 4. Default (8000)

    if args.port:
        port = args.port
        source = "Flag"
    elif os.environ.get("AGENT_PORT"):  # <--- The fix for AgentBeats
        port = int(os.environ.get("AGENT_PORT"))
        source = "Env Var (AGENT_PORT)"
    elif os.environ.get("PORT"):
        port = int(os.environ.get("PORT"))
        source = "Env Var (PORT)"
    else:
        port = 8000
        source = "Default"

    print(f"🚀 Green Agent Controller Launching...")
    print(f"   - Detected Port: {port} (Source: {source})")
    
    uvicorn.run(app, host="0.0.0.0", port=port)