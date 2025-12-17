# import os
# import uvicorn
# import requests
# import json
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel

# # Import your existing Green Agent
# from src.green_agent.green_agent import GreenAgent

# # --- 1. REMOTE ADAPTER ---
# class RemoteAgentAdapter:
#     """
#     Connects the Green Agent to a remote White Agent (A2A over HTTP).
#     """
#     def __init__(self, target_url):
#         self.target_url = target_url.rstrip('/')
#         if not self.target_url.endswith("/agent/tasks"):
#              self.target_url += "/agent/tasks"

#     def receive_task(self, message, image_path=None):
#         payload = {
#             "message": message,
#             "image_path": image_path if image_path else None 
#         }
#         try:
#             # We send the path to the white agent. 
#             # Since we are running locally, the absolute path works for both processes.
#             response = requests.post(self.target_url, json=payload, timeout=60)
#             response.raise_for_status()
#             return response.json()
#         except Exception as e:
#             return {"error": f"Remote Connection Failed: {str(e)}"}

# # --- 2. FASTAPI SERVER ---
# app = FastAPI()

# # MOUNT STATIC FILES
# # 1. Output: So you can see the HTML report in your browser
# # 2. Dataset: So the HTML report can load the images (fixes broken image icons)
# os.makedirs("output", exist_ok=True)
# app.mount("/report", StaticFiles(directory="output"), name="report")

# # Check if dataset exists before mounting to avoid crash
# if os.path.exists("dataset"):
#     app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")
# else:
#     print("⚠️ WARNING: 'dataset' folder not found in current directory.")

# # Initialize Agent
# green_agent = GreenAgent(model_name="llama3.2")

# class AssessmentRequest(BaseModel):
#     message: str

# @app.post("/agent/tasks")
# async def run_assessment_task(req: AssessmentRequest, request: Request):
#     """
#     AgentBeats Entry Point.
#     Receives a request to run an assessment against a target URL.
#     """
#     prompt = req.message
    
#     # Extract URL from prompt "Assess http://..."
#     words = prompt.split()
#     target_url = next((w for w in words if w.startswith("http")), None)
    
#     if not target_url:
#         return {"status": "error", "result": "Could not find a valid http(s) URL in your request."}

#     print(f"🔗 Connecting to Remote Agent at: {target_url}")
    
#     # Connect to the remote agent
#     adapter = RemoteAgentAdapter(target_url)
#     green_agent.connect_white_agent(adapter)
    
#     # CRITICAL FIX: Use Absolute Path
#     # Using os.path.abspath ensures the White Agent (running in a different process)
#     # can find the image file on the disk.
#     dataset_path = os.path.abspath(os.path.join(os.getcwd(), "dataset"))
    
#     if not os.path.exists(dataset_path):
#         return {"status": "error", "result": "Dataset not found in container."}

#     try:
#         # Run Assessment
#         result = green_agent.run_assessment(dataset_path, limit=5, agent_name="Remote_Candidate")
        
#         # Generate HTML Artifacts
#         output_dir = os.path.join(os.getcwd(), "output")
#         green_agent.generate_artifacts(output_dir)
        
#         # Build the Clickable Link
#         base_url = str(request.base_url).rstrip("/")
#         report_link = f"{base_url}/report/leaderboard.html"
        
#         return {
#             "status": "success",
#             "verdict": result.get('overall_grade'),
#             "score": result.get('overall_score_percent'),
#             "analysis": result.get('analysis'),
#             "view_report_here": report_link
#         }
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return {"status": "error", "result": str(e)}

# # --- 3. AGENT IDENTITY CARD ---
# @app.get("/.well-known/agent-card.json")
# def agent_card():
#     return JSONResponse(content={
#         "name": "AutoDrive Assessment Agent",
#         "description": "Evaluates driving agents on safety and planning.",
#         "version": "1.0.0",
#         "capabilities": ["assessment", "vision"],
#         "author": "Vihaal"
#     })

# @app.get("/health")
# def health_check():
#     return {"status": "operational"}

# if __name__ == "__main__":
#     host = os.getenv("HOST", "0.0.0.0")
#     port = int(os.getenv("AGENT_PORT", 8010))
#     uvicorn.run(app, host=host, port=port)


import os
import uvicorn
import requests
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
        # Ensure we don't double-append "/agent/tasks" if the user already provided it
        if not self.target_url.endswith("/tasks"):
             self.target_url += "/tasks" # Most basic append, assuming standard structure

    def receive_task(self, message, image_path=None):
        payload = {
            "message": message,
            "image_path": image_path if image_path else None 
        }
        try:
            # Send the absolute path to the white agent
            print(f"    -> Sending task to {self.target_url}")
            response = requests.post(self.target_url, json=payload, timeout=5000)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"    ❌ Remote Connection Failed: {e}")
            return {"error": f"Remote Connection Failed: {str(e)}"}

# --- 2. FASTAPI SERVER ---
app = FastAPI()

# MOUNT STATIC FILES (Critical for HTML Report)
os.makedirs("output", exist_ok=True)
app.mount("/report", StaticFiles(directory="output"), name="report")

if os.path.exists("dataset"):
    app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

# Initialize Agent
try:
    green_agent = GreenAgent(model_name="llama3.2")
    print("✅ Judge (Green Agent) initialized.")
except Exception as e:
    print(f"⚠️ Green Agent Init Failed: {e}")
    green_agent = None

class AssessmentRequest(BaseModel):
    message: str

# --- 3. IDENTITY CARD (Fixes 0/1) ---
@app.get("/.well-known/agent-card.json")
def agent_card():
    return JSONResponse(content={
        "name": "AutoDrive Judge",
        "description": "Evaluates driving agents on safety and planning.",
        "version": "1.0.0",
        "capabilities": ["assessment", "vision"],
        "author": "User"
    })

@app.post("/agent/tasks")
async def run_assessment_task(req: AssessmentRequest, request: Request):
    """
    The Real Logic: Parses URL, Connects, Runs Assessment, Generates Report.
    """
    if not green_agent:
        return {"status": "error", "result": "Judge model is not loaded."}

    prompt = req.message
    
    # Extract URL from prompt "Assess the agent at http://..."
    words = prompt.split()
    target_url = next((w for w in words if w.startswith("http")), None)
    
    if not target_url:
        return {"status": "error", "result": "Could not find a valid http(s) URL in your request."}

    print(f"🔗 Connecting to Remote Agent at: {target_url}")
    
    # Connect to the remote agent
    adapter = RemoteAgentAdapter(target_url)
    green_agent.connect_white_agent(adapter)
    
    # Use Absolute Path for Dataset
    dataset_path = os.path.abspath(os.path.join(os.getcwd(), "dataset"))
    
    if not os.path.exists(dataset_path):
        return {"status": "error", "result": "Dataset folder not found."}

    try:
        # Run Assessment (Limit to 5 for speed)
        print("🚦 Starting Assessment Loop...")
        result = green_agent.run_assessment(dataset_path, limit=2, agent_name="Remote_Candidate")
        
        # Generate HTML Artifacts
        output_dir = os.path.join(os.getcwd(), "output")
        green_agent.generate_artifacts(output_dir)
        
        # Build the Clickable Link
        base_url = str(request.base_url).rstrip("/")
        report_link = f"{base_url}/report/leaderboard.html"
        
        print(f"✅ Assessment Complete! Score: {result.get('overall_score_percent')}%")
        
        return {
            "status": "success",
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
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", 8010))
    uvicorn.run(app, host=host, port=port)