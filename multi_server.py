import uvicorn
from fastapi import FastAPI, Request
from src.white_agent.white_agent import WhiteAgent

app = FastAPI()

# Initialize our fleet of agents
agents = {
    "minicpm-v": WhiteAgent(model_name="minicpm-v"),
    "bakllava": WhiteAgent(model_name="bakllava"),
    "llava": WhiteAgent(model_name="llava"),
    "moondream": WhiteAgent(model_name="moondream"),
    "llama": WhiteAgent(model_name="llama3.2"),
    "mock": WhiteAgent(model_name="mock") # Always passes
}

@app.post("/agent/{agent_name}/tasks")
async def route_task(agent_name: str, request: Request):
    """
    Dynamic Router: Sends the task to the specific agent requested in the URL.
    """
    if agent_name not in agents:
        return {"error": f"Agent '{agent_name}' not found. Available: {list(agents.keys())}"}
    
    data = await request.json()
    message = data.get("message", "")
    image_path = data.get("image_path")
    
    print(f"🔀 Routing task to Agent: [{agent_name.upper()}]")
    
    # Delegate to the specific agent instance
    return agents[agent_name].receive_task(message, image_path)

if __name__ == "__main__":
    print("🤖 Multi-Agent Server Running on Port 8001")
    print("   Endpoints available:")
    print("   - http://127.0.0.1:8001/agent/moondream/tasks")
    print("   - http://127.0.0.1:8001/agent/mock/tasks")
    uvicorn.run(app, host="0.0.0.0", port=8001)