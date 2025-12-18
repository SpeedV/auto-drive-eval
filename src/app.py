import os
from src.green_agent.green_agent import GreenAgent
from src.white_agent.white_agent import WhiteAgent

class AutoDriveBenchmark:
    def __init__(self):
        print("🔌 Initializing AutoDrive Agents (OpenAI Mode)...")
        # Ensure we use GPT models
        self.white_agent = WhiteAgent(model_name="gpt-4o-mini")
        self.green_agent = GreenAgent(model_name="gpt-4o-mini")
        self.green_agent.connect_white_agent(self.white_agent)

    def run_benchmark(self):
        dataset_path = os.path.abspath("dataset")
        if not os.path.exists(dataset_path):
            print("❌ Dataset not found!")
            return
            
        print("🚀 Starting Assessment...")
        self.green_agent.run_assessment(
            dataset_path=dataset_path, 
            limit=5, 
            agent_name="GPT-4o-Driver"
        )
        
        # Generate Report
        output_dir = "results"
        html_path = self.green_agent.generate_artifacts(output_dir)
        print(f"🏁 Benchmark Complete! Report: {html_path}")