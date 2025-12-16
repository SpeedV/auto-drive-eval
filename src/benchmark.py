import os
import json
import sys
import random
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_loader import SplitFolderDataset
from white_agent import WhiteAgent
from green_agent import GreenAgent
from html_reporter import generate_leaderboard_report

def run_benchmark():
    # --- CONFIGURATION ---
    # Set to None to run the full dataset (All Train / All Test)
    # Set to integer for limited run
    LIMIT = 5  
    
    # How many training examples to use if LIMIT is active.
    # (If LIMIT is None, we ignore this and use ALL training data)
    MAX_TRAIN_EXAMPLES_IF_LIMITED = 20
    
    # True = Pick random examples. False = Pick first N (Deterministic).
    RANDOMIZE_TRAIN = True         
    
    MODELS = [
        # "llama3.2-vision", 
        "moondream", 
        "moondream-cautious",    
        "moondream-aggressive",  
        "minicpm-v",
        "llava",
        "bakllava",            
    ]
    
    ROOT_DIR = os.getcwd() 
    DATASET_PATH = os.path.join(ROOT_DIR, "dataset")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    JSON_FILE = os.path.join(OUTPUT_DIR, "tournament_results.json")
    HTML_FILE = os.path.join(OUTPUT_DIR, "leaderboard.html")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        dataset = SplitFolderDataset(root_dir=DATASET_PATH, split_ratio=0.8)
    except FileNotFoundError as e:
        print(e)
        return

    print("👨‍⚖️ Initializing Green Agent (Judge)...")
    green_agent = GreenAgent(model_name="llama3.2")
    tournament_results = {}

    # --- DATA SELECTION LOGIC ---
    
    # 1. Test Data
    all_test_data = dataset.get_test_data()
    if LIMIT is None:
        print(f"🚀 LIMIT is None: Using ALL {len(all_test_data)} test images.")
        test_data = all_test_data
    elif LIMIT < len(all_test_data):
        print(f"🎲 Selecting {LIMIT} random images for the shared test set...")
        random.seed(None) 
        test_data = random.sample(all_test_data, LIMIT)
    else:
        test_data = all_test_data

    # 2. Training Data (Shared across all models)
    all_train_data = dataset.get_train_data()
    
    if LIMIT is None:
        print(f"🚀 LIMIT is None: Using ALL {len(all_train_data)} training examples.")
        train_examples = all_train_data
    else:
        # We are in limited mode, apply cap
        if len(all_train_data) > MAX_TRAIN_EXAMPLES_IF_LIMITED:
            if RANDOMIZE_TRAIN:
                print(f"🎲 Randomizing Training Data (Subset of {MAX_TRAIN_EXAMPLES_IF_LIMITED})...")
                train_examples = random.sample(all_train_data, MAX_TRAIN_EXAMPLES_IF_LIMITED)
            else:
                print(f"🔒 Deterministic Training Data (First {MAX_TRAIN_EXAMPLES_IF_LIMITED})...")
                train_examples = all_train_data[:MAX_TRAIN_EXAMPLES_IF_LIMITED]
        else:
            train_examples = all_train_data

    # --- START TOURNAMENT ---
    print(f"\n🏆 STARTING 5-AGENT TOURNAMENT 🏆")
    
    model_pbar = tqdm(MODELS, desc="Tournament Progress", unit="model")
    
    for model_name in model_pbar:
        model_pbar.set_description(f"🤖 Agent: {model_name}")
        
        try:
            agent = WhiteAgent(model_name=model_name)
        except Exception as e:
            print(f"\nSkipping {model_name}: {e}")
            continue

        agent.train(train_examples)
        model_details = []
        
        img_pbar = tqdm(enumerate(test_data), total=len(test_data), desc="Testing", leave=False, unit="img")
        for i, (image_path, context, goal, ground_truth) in img_pbar:
            img_id = ground_truth.get('id', i)
            img_pbar.set_description(f"Processing Img {img_id}")

            start_ts = time.time()
            agent_response = agent.generate_response(image_path, context, goal)
            duration = round(time.time() - start_ts, 2)

            eval_report = green_agent.evaluate(agent_response, ground_truth)
            eval_report['latency'] = duration
            model_details.append(eval_report)

        model_pbar.set_description(f"📝 Compiling Analysis: {model_name}")
        final_analysis = green_agent.compile_final_report(model_details, model_name)
        
        tournament_results[model_name] = {
            "analysis": final_analysis,
            "details": model_details
        }

    print("\n🏁 Generating Interactive Dashboard...")
    with open(JSON_FILE, 'w') as f:
        json.dump(tournament_results, f, indent=4)
    generate_leaderboard_report(JSON_FILE, HTML_FILE)
    print(f"📊 LEADERBOARD READY: {HTML_FILE}")

if __name__ == "__main__":
    run_benchmark()


