# import os
# import uvicorn
# from starlette.staticfiles import StaticFiles

# # Official A2A SDK Imports
# from a2a.server.apps import A2AStarletteApplication
# from a2a.server.request_handlers import DefaultRequestHandler
# from a2a.server.agent_execution import AgentExecutor, RequestContext
# from a2a.server.events import EventQueue
# from a2a.server.tasks import InMemoryTaskStore
# from a2a.types import AgentCard, AgentCapabilities, AgentSkill
# from a2a.utils import new_agent_text_message

# from src.green_agent.green_agent import GreenAgent

# # --- 1. EXECUTOR ---
# class AutoDriveExecutor(AgentExecutor):
#     def __init__(self):
#         self.green = GreenAgent(model_name="gpt-4o-mini")

#     async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
#         user_message = context.get_user_input()
#         print(f"📨 Received Command: {user_message}")

#         # Initial Status
#         await event_queue.enqueue_event(new_agent_text_message("🚦 Starting AutoDrive Assessment..."))

#         try:
#             dataset_path = os.path.join(os.getcwd(), "dataset")
#             result = self.green.run_assessment(dataset_path, limit=5, agent_name="GPT-4o-Driver")
            
#             # Use dynamic port for link
#             port = os.environ.get("AGENT_PORT", "8001")
#             report_link = f"http://localhost:{port}/results/leaderboard.html"
            
#             score = result.get('overall_score_percent', 0)
#             verdict = result.get('overall_grade', 'N/A')

#             response_text = (
#                 f"✅ **Assessment Complete!**\n"
#                 f"- **Verdict:** {verdict}\n"
#                 f"- **Score:** {score}%\n\n"
#                 f"[📊 Click Here to View Report]({report_link})"
#             )
#             await event_queue.enqueue_event(new_agent_text_message(response_text))
#             print(f"✅ Finished. Score: {score}%")

#         except Exception as e:
#             error_msg = f"❌ Error: {str(e)}"
#             print(error_msg)
#             import traceback
#             traceback.print_exc()
#             await event_queue.enqueue_event(new_agent_text_message(error_msg))

#     async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
#         print(f"🛑 Cancel requested for context {context.context_id}")
#         await event_queue.enqueue_event(new_agent_text_message("🛑 Task cancelled."))

# # --- 2. ENTRY POINT ---
# if __name__ == "__main__":
#     host = os.environ.get("HOST", "0.0.0.0")
#     port = int(os.environ.get("AGENT_PORT", "8001"))

#     # Agent Card
#     agent_card = AgentCard(
#         name="AutoDrive Green Agent",
#         description="Autonomous Driving Benchmark Evaluator",
#         version="1.0.0",
#         url=f"http://{host}:{port}",
#         capabilities=AgentCapabilities(streaming=True),
#         default_input_modes=["text"],
#         default_output_modes=["text"],
#         skills=[
#             AgentSkill(
#                 id="eval_drive",
#                 name="Evaluate Driving Agent",
#                 description="Runs the autonomous driving safety benchmark",
#                 tags=["benchmark", "safety"],
#                 input_mode="text",
#                 output_mode="text"
#             )
#         ]
#     )

#     # Initialize SDK App Wrapper
#     a2a_wrapper = A2AStarletteApplication(
#         agent_card=agent_card,
#         http_handler=DefaultRequestHandler(
#             agent_executor=AutoDriveExecutor(),
#             task_store=InMemoryTaskStore(),
#         ),
#     )

#     # [CRITICAL FIX]
#     # The SDK wrapper creates a Starlette app internally. We must access it to mount files.
#     # The internal Starlette app is usually at .app
#     main_app = a2a_wrapper.build() 

#     # Mount the 'output' folder for HTML reports
#     os.makedirs("output", exist_ok=True)
#     main_app.mount("/results", StaticFiles(directory="output"), name="results")

#     print(f"🚀 AutoDrive Agent launching on {host}:{port}")
    
#     # Run the underlying Starlette app, NOT the wrapper object
#     uvicorn.run(main_app, host=host, port=port)


import os
import uvicorn
import json
from starlette.staticfiles import StaticFiles

# Official A2A SDK Imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message

# Import both agents
try:
    from src.green_agent.green_agent import GreenAgent
    from src.white_agent.white_agent import WhiteAgent
except ImportError:
    from green_agent import GreenAgent
    from white_agent import WhiteAgent

# --- WHITE AGENT (THE DRIVER) ---
class WhiteDriverExecutor(AgentExecutor):
    def __init__(self):
        self.agent = WhiteAgent(model_name="gpt-4o-mini")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_message = context.get_user_input()
        print(f"⬜ White Agent Task: {user_message[:50]}...")
        
        response_data = self.agent.receive_task(user_message)
        
        await event_queue.enqueue_event(
            new_agent_text_message(json.dumps(response_data, indent=2))
        )

    # --- FIXED: ADDED REQUIRED CANCEL METHOD ---
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        print("⬜ White Agent Cancel Requested")
        await event_queue.enqueue_event(new_agent_text_message("🛑 Task cancelled by user."))

def create_white_app(public_url):
    print("⚪ Initializing White Agent Mode")
    
    skill = AgentSkill(
        id="drive", 
        name="Drive", 
        description="Driving logic", 
        input_mode="text", 
        output_mode="text",
        tags=["automotive", "driving"] 
    )
    
    card = AgentCard(
        name="AutoDrive White Agent",
        description="Autonomous Vehicle AI Target",
        url=public_url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        default_input_modes=["text"],
        default_output_modes=["text"]
    )
    return A2AStarletteApplication(
        agent_card=card,
        http_handler=DefaultRequestHandler(agent_executor=WhiteDriverExecutor(), task_store=InMemoryTaskStore()),
    ).build()

# --- GREEN AGENT (THE JUDGE) ---
class GreenJudgeExecutor(AgentExecutor):
    def __init__(self):
        self.green = GreenAgent(model_name="gpt-4o-mini")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_message = context.get_user_input()
        print(f"🚦 Green Agent Command: {user_message}")
        await event_queue.enqueue_event(new_agent_text_message("🚦 Starting Assessment..."))

        try:
            dataset_path = os.path.join(os.getcwd(), "dataset")
            self.green.run_assessment(dataset_path, limit=5, agent_name="GPT-4o-Driver")
            
            report_url = f"{os.getenv('AGENT_URL')}/results/leaderboard.html"
            await event_queue.enqueue_event(new_agent_text_message(f"✅ Done. [View Report]({report_url})"))
        except Exception as e:
            await event_queue.enqueue_event(new_agent_text_message(f"❌ Error: {str(e)}"))

    # --- FIXED: ADDED REQUIRED CANCEL METHOD ---
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        print("🚦 Green Agent Cancel Requested")
        await event_queue.enqueue_event(new_agent_text_message("🛑 Assessment cancelled."))

def create_green_app(public_url):
    print("🟢 Initializing Green Agent Mode")
    
    skill = AgentSkill(
        id="eval", 
        name="Evaluate", 
        description="Runs benchmark", 
        input_mode="text", 
        output_mode="text",
        tags=["benchmark", "safety"]
    )
    
    card = AgentCard(
        name="AutoDrive Green Agent",
        description="Benchmark Evaluator",
        url=public_url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        default_input_modes=["text"],
        default_output_modes=["text"]
    )
    
    wrapper = A2AStarletteApplication(
        agent_card=card,
        http_handler=DefaultRequestHandler(agent_executor=GreenJudgeExecutor(), task_store=InMemoryTaskStore()),
    )
    app = wrapper.build()
    os.makedirs("output", exist_ok=True)
    app.mount("/results", StaticFiles(directory="output"), name="results")
    return app

# --- MAIN SWITCH ---
if __name__ == "__main__":
    import sys

    # 1. Listen on the port Cloud Run/Controller expects
    # The Controller sets AGENT_PORT. Local testing defaults to 8001.
    port = int(os.environ.get("AGENT_PORT", 8001))
    host = "0.0.0.0"
    
    # 2. Determine Role
    role = os.environ.get("ROLE", "green").lower()
    
    # 3. GET THE PUBLIC URL (The Fix)
    # We prioritize AGENT_URL (set by us in Cloud Run).
    # If that's missing, we check CLOUDRUN_HOST (set by Controller).
    # Fallback to local only if absolutely necessary.
    if os.environ.get("AGENT_URL"):
        public_url = os.environ.get("AGENT_URL")
    elif os.environ.get("CLOUDRUN_HOST"):
        public_url = f"https://{os.environ.get('CLOUDRUN_HOST')}"
    else:
        public_url = f"http://{host}:{port}"

    print(f"🔌 Starting {role.upper()} Agent on port {port}")
    print(f"🌍 Advertising Public URL: {public_url}")

    if role == "white":
        app = create_white_app(public_url)
    else:
        app = create_green_app(public_url)

    uvicorn.run(app, host=host, port=port)