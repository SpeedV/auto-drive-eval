# import argparse
# import os
# import uvicorn
# import requests
# import json
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel

# # Import BOTH Agents
# from src.green_agent.green_agent import GreenAgent
# from src.white_agent.white_agent import WhiteAgent  # <--- ADDED THIS IMPORT

# # --- 1. REMOTE ADAPTER ---
# class RemoteAgentAdapter:
#     """
#     Connects the Green Agent to a remote White Agent (A2A over HTTP).
#     """
#     def __init__(self, target_url):
#         self.target_url = target_url.rstrip('/')

#     def receive_task(self, message, image_path=None):
#         payload = {
#             "message": message,
#             "image_path": image_path if image_path else None 
#         }
#         try:
#             # Send task to remote agent
#             response = requests.post(self.target_url, json=payload, timeout=60)
#             response.raise_for_status()
#             return response.json()
#         except Exception as e:
#             return {"error": f"Remote Connection Failed: {str(e)}"}

# # --- 2. FASTAPI SERVER ---
# app = FastAPI()

# # MOUNT STATIC FILES
# output_dir = os.path.join(os.getcwd(), "output")
# dataset_dir = os.path.join(os.getcwd(), "dataset")
# os.makedirs(output_dir, exist_ok=True)

# app.mount("/results", StaticFiles(directory=output_dir), name="results")
# if os.path.exists(dataset_dir):
#     app.mount("/dataset", StaticFiles(directory=dataset_dir), name="dataset")

# # GLOBAL STATE
# green = GreenAgent(model_name="gpt-4o-mini") 

# # Port Global for Link Generation
# port = 8010 

# class TaskRequest(BaseModel):
#     message: str
#     target_url: str = None  # Optional

# @app.post("/agent/tasks")
# async def handle_task(req: TaskRequest):
#     """
#     Trigger the Green Agent to run a benchmark.
#     """
#     print(f"📨 Received Command: {req.message}")
    
#     # --- FIX IS HERE ---
#     if req.target_url:
#         # Case A: Connect to a Remote Agent (if URL provided)
#         print(f"🔌 Connecting to Remote White Agent at: {req.target_url}")
#         adapter = RemoteAgentAdapter(req.target_url)
#         green.connect_white_agent(adapter)
#     else:
#         # Case B: Use LOCAL White Agent (Default)
#         # This fixes the "Failed to connect" error by running the agent in-process
#         print(f"🏠 Using Local White Agent (In-Process)")
#         local_white = WhiteAgent(model_name="gpt-4o-mini")
#         green.connect_white_agent(local_white)
#     # -------------------

#     # Run Assessment
#     try:
#         agent_name = "GPT-4o-Driver"
#         dataset_path = os.path.join(os.getcwd(), "dataset")
        
#         # Run the Green Agent Logic
#         result = green.run_assessment(dataset_path, limit=5, agent_name=agent_name)
        
#         # Generate HTML Report
#         html_path = green.generate_artifacts(output_dir)
        
#         # Dynamic Link Generation
#         report_link = f"http://localhost:{port}/results/leaderboard.html"
        
#         print(f"✅ Assessment Complete! Score: {result.get('overall_score_percent')}%")
#         print(f"📊 REPORT READY: {report_link}") 
        
#         return {
#             "status": "success",
#             "agent": agent_name,
#             "verdict": result.get('overall_grade'),
#             "score": result.get('overall_score_percent'),
#             "view_report_here": report_link
#         }
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return {"status": "error", "result": str(e)}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, help="Port to run the server on")
#     args, unknown = parser.parse_known_args()

#     if args.port: port = args.port
#     elif os.environ.get("AGENT_PORT"): port = int(os.environ.get("AGENT_PORT"))
#     elif os.environ.get("PORT"): port = int(os.environ.get("PORT"))
#     else: port = 8010

#     print(f"🚀 Green Agent Server running on port {port}")
#     uvicorn.run(app, host="0.0.0.0", port=port)



import argparse
import os
import uvicorn
import requests
import json
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import Agents (Assumes 'src' folder is in the same directory)
from src.green_agent.green_agent import GreenAgent
from src.white_agent.white_agent import WhiteAgent

# --- 1. REMOTE ADAPTER ---
class RemoteAgentAdapter:
    def __init__(self, target_url):
        self.target_url = target_url.rstrip('/')

    def receive_task(self, message, image_path=None):
        payload = {"message": message, "image_path": image_path if image_path else None}
        try:
            response = requests.post(self.target_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Remote Connection Failed: {str(e)}"}

# --- 2. FASTAPI SERVER ---
app = FastAPI()

# MOUNT STATIC FILES
# We use os.path.join(os.getcwd(), ...) to be safe with relative paths
output_dir = os.path.join(os.getcwd(), "output")
dataset_dir = os.path.join(os.getcwd(), "dataset")
os.makedirs(output_dir, exist_ok=True)

app.mount("/results", StaticFiles(directory=output_dir), name="results")
if os.path.exists(dataset_dir):
    app.mount("/dataset", StaticFiles(directory=dataset_dir), name="dataset")

# GLOBAL STATE
green = GreenAgent(model_name="gpt-4o-mini") 
port = 8010 # Default placeholder

# --- 3. AGENT CARD (REQUIRED FOR AGENTBEATS) ---
@app.get("/.well-known/agent-card.json")
def agent_card():
    return JSONResponse({
        "name": "AutoDrive Green Agent",
        "description": "A2A Evaluation Agent for Autonomous Driving",
        "version": "1.0.0",
        "framework": "fastapi",
        "author": "Vihaal"
    })

class TaskRequest(BaseModel):
    message: str
    target_url: str = None 

@app.post("/agent/tasks")
async def handle_task(req: TaskRequest):
    print(f"📨 Received Command: {req.message}")
    
    # 1. Connect White Agent
    if req.target_url:
        print(f"🔌 Connecting to Remote White Agent at: {req.target_url}")
        adapter = RemoteAgentAdapter(req.target_url)
        green.connect_white_agent(adapter)
    else:
        # Default to Local In-Process White Agent (Fixes connection errors)
        print(f"🏠 Using Local White Agent (In-Process)")
        local_white = WhiteAgent(model_name="gpt-4o-mini")
        green.connect_white_agent(local_white)

    # 2. Run Assessment
    try:
        agent_name = "GPT-4o-Driver"
        dataset_path = os.path.join(os.getcwd(), "dataset")
        
        result = green.run_assessment(dataset_path, limit=5, agent_name=agent_name)
        html_path = green.generate_artifacts(output_dir)
        
        # Link generation uses the dynamic 'port' variable
        report_link = f"http://localhost:{port}/results/leaderboard.html"
        
        print(f"✅ Assessment Complete! Score: {result.get('overall_score_percent')}%")
        print(f"📊 REPORT READY: {report_link}") 
        
        return {
            "status": "success",
            "agent": agent_name,
            "verdict": result.get('overall_grade'),
            "score": result.get('overall_score_percent'),
            "view_report_here": report_link
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "result": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    import sys

    # 1. Get the configuration from the AgentBeats Controller
    #    The controller injects these variables automatically when it runs your agent.
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("AGENT_PORT", "8001")

    # 2. Print startup info so we can see it in the logs
    print(f"🚀 Agent launching on {host}:{port}")

    # 3. Start the Server
    #    Ensure 'app' is the name of your FastAPI/Flask instance above
    uvicorn.run(app, host=host, port=int(port))