# AutoDrive Adversarial Benchmark

A rigorous "LLM-as-a-Judge" benchmark for evaluating Vision-Language Models (VLMs) on autonomous driving tasks. This framework uses a **Green Agent** (Judge) to evaluate **White Agents** (Driver Models) on Perception, Prediction, and Planning accuracy, while enforcing strict U.S. Traffic Laws via a deterministic Rules Engine.

## Design

1.  **White Agent (The Student):** Receives an image + context. Outputs a driving plan.
    * *Supported Models:* Llama 3.2 Vision, Moondream, LLaVA, BakLLaVA, MiniCPM-V.
2.  **Green Agent (The Judge):**
    * **LLM Grader:** Uses Llama 3.2 (Text) with `seed=42` to deterministically score logic (0.0 to 1.0).
    * **Rules Engine:** A Python-based legal auditor that applies hard penalties for specific traffic violations (e.g., passing on double yellow lines).

---

## Getting Started

### 1. Prerequisites
* **Python 3.10+**
* **Ollama** installed and running (`ollama serve`).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Repo Structure

```bash
auto-drive-eval/
├── dataset/
│   ├── images/         # datset images here
│   └── descriptions/   # dataset 'ground truth' descriptions here
├── src/
│   ├── benchmark.py
│   ├── dataset_loader.py
│   ├── green_agent.py
│   ├── white_agent.py
│   ├── rules_engine.py
│   ├── html_reporter.py
│   └── test_green_agent.py
├── output/             
├── requirements.txt
└── README.md
```

## Required Ollama Models

### The Judge
ollama pull llama3.2

### The Competitors
ollama pull llama3.2-vision
ollama pull moondream
ollama pull minicpm-v
ollama pull llava
ollama pull bakllava

## Running the Benchmark

To run the full tournament (evaluating all models against the test set):
```bash
python src/benchmark.py
```

### Benchmark Process:
1.  Loads training set (or randomized set of size LIMIT*2 if LIMIT is set) 
    * Regardless of LIMIT size the minimum is size 20 training set to ensure White Agents have appropriate ammount of training data
2.  Iterates through all models
    * Trains model using trianing set (with ground-truth shown)
    * Tests model on test set (with ground-truth hidden, only showing image, context, and goal)
    * Stores White Agent's responses
3. Generates detailed logs in the console with a progress bar
4. Compiles a final HTML Leaderboard and Comprehensive Report

### Configuration:
You can adjust settings in src/benchmark.py:
* LIMIT: Number of test images to evaluate (Default: 5).
* RANDOMIZE_TRAIN: Toggle True/False to randomize the Few-Shot examples.

## Green-Agent Evaluation:

To ensure the Green Agent is grading fairly and accurately, run the validation suite. This runs 50 specific edge cases (e.g., "Ambulance Blocking", "School Zone Speeding") where the scores are known in advance.

```bash
python src/test_green_agent.py
```

### Test Criteria:
* ✅ PASS: The Green Agent correctly identifies the violation or safe behavior.
* ❌ FAIL: The Green Agent missed a violation or penalized safe driving.

## Reproducing Results:

This benchmark ensures Deterministic Reproduction to verify findings:
* Open src/benchmark.py.
* Set RANDOMIZE_TRAIN = False. This forces the data loader to pick the exact same training examples every time.
* Set LIMIT = to a fixed number to run on a fixed set of test images.
* Run: python src/benchmark.py

Because the Green Agent uses seed=42 and temperature=0, running this command multiple times will yield identical scores for the same inputs, ensuring reproducability

## Metrics:

* Perception Score: Did the white agent list all safety-critical objects (Traffic lights, Pedestrians)?
* Prediction Score: Did the white agent correctly anticipate future movement? (In comparision to 'ground-truth')
* Planning Score: Is the white agent's driving decision and plan safe and legal? (especially in accordance to US road laws)
* Safety Violations: A count of specific legal infractions (e.g., "Running a Red Light").
    * Note: Any safety violation results in a massive penalty to the Planning Score since such safety violations are not only illegal but also highly unsafe, which is against the purpose of these white agents.


