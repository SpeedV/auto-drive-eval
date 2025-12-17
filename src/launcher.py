import os
import sys
import argparse
import time

# Ensure we can find the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from green_agent.green_agent import GreenAgent
from white_agent.white_agent import WhiteAgent

def main():
    parser = argparse.ArgumentParser(description="AutoDrive Agentified Tournament")
    parser.add_argument("--models", nargs='+', default=["moondream", "llava"], 
                        help="List of Ollama models to test")
    parser.add_argument("--limit", type=int, default=5, help="Number of test cases per model")
    args = parser.parse_args()

    print("\n" + "="*60)
    print(f"🚦 STARTING AGENTIFIED ASSESSMENT")
    print(f"MODELS: {args.models}")
    print(f"TEST LIMIT: {args.limit}")
    print(f"TRAIN POOL: >= {max(args.limit * 2, 20)} items (Constraint)")
    print("="*60 + "\n")

    print("👨‍⚖️ Initializing Green Agent...")
    green = GreenAgent(model_name="llama3.2")
    
    dataset_path = os.path.join(os.getcwd(), "dataset")

    for model_name in args.models:
        print(f"\n🤖 Round Starting: {model_name}")
        
        try:
            white = WhiteAgent(model_name=model_name)
            green.connect_white_agent(white)
            
            # Pass limit to run_assessment, which now handles splitting
            result = green.run_assessment(
                dataset_path, 
                limit=args.limit, 
                agent_name=model_name
            )
            
            if not result:
                print("   ⚠️ No results generated.")
                continue

            metrics = result.get('metrics', {})
            score = result.get('overall_score_percent', 0)
            grade = result.get('overall_grade', 'N/A')
            violations = metrics.get('total_violations', 0)
            
            print(f"   Verdict: {grade} | Score: {score}% | Violations: {violations}")

        except Exception as e:
            print(f"   ❌ Skipped {model_name} due to error: {e}")
            continue

    print("\n" + "="*60)
    print("🏁 TOURNAMENT COMPLETE")
    
    output_dir = os.path.join(os.getcwd(), "output")
    try:
        html_file = green.generate_artifacts(output_dir)
        print(f"📊 LEADERBOARD GENERATED: {html_file}")
    except Exception as e:
        print(f"❌ Failed to generate report: {e}")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    main()