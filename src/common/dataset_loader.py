import os
import json
import random

class SplitFolderDataset:
    def __init__(self, root_dir, seed=42):
        self.root_dir = os.path.abspath(root_dir)
        self.images_dir = os.path.join(self.root_dir, "images")
        self.desc_dir = os.path.join(self.root_dir, "descriptions")
        self.seed = seed
        
        # Handle path resolution
        if not os.path.exists(self.images_dir):
             base = os.getcwd()
             self.images_dir = os.path.join(base, "dataset", "images")
             self.desc_dir = os.path.join(base, "dataset", "descriptions")

        if not os.path.exists(self.images_dir):
            print(f"⚠️ Warning: {self.images_dir} not found.")
            self.all_files = []
        else:
            self.all_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))])
            
        # --- FIXED LOGIC: DETERMINISTIC HARD SPLIT ---
        # We keep this part strictly deterministic so "Test" images never leak into "Train"
        # regardless of how we sample them later.
        rng_split = random.Random(self.seed) 
        rng_split.shuffle(self.all_files)
        
        # 2. Hard Split: 125 Training (Source) / 75 Test (Source)
        split_point = 125
        self.source_train = self.all_files[:split_point]
        self.source_test = self.all_files[split_point:]
        
        # Runtime buckets
        self.active_train_pool = []
        self.active_test_batch = []

    def prepare_runtime_buckets(self, test_limit, seed=None):
        """
        Populates buckets based on the requested test limit.
        Args:
            test_limit: Number of test cases to run.
            seed: If None, picks NEW random images every time. 
                  If set (e.g. 123), picks the SAME random images every time.
        """
        # We use a local Random instance so we don't mess with global state
        rng_runtime = random.Random(seed)

        # 1. Select Test Cases (from Test Source)
        # Instead of slicing [0:limit], we SAMPLE randomly from the 75 test files.
        available_test_files = list(self.source_test) # Copy to be safe
        
        if not test_limit or test_limit > len(available_test_files):
            self.active_test_batch = available_test_files
            rng_runtime.shuffle(self.active_test_batch)
        else:
            self.active_test_batch = rng_runtime.sample(available_test_files, test_limit)
            
        # 2. Select Training Pool (from Train Source)
        # We keep the pool logic simple (taking the first N), as get_few_shot_examples
        # handles the randomness for context learning separately.
        eff_limit = test_limit if test_limit else len(self.source_test)
        needed_train_size = max(20, 2 * eff_limit)
        
        needed_train_size = min(needed_train_size, len(self.source_train))
        self.active_train_pool = self.source_train[:needed_train_size]

        print(f"   📂 Dataset Loaded. Split: {len(self.source_train)} Train / {len(self.source_test)} Test")
        
        mode_msg = f"Deterministic (Seed {seed})" if seed is not None else "Random (New Shuffle)"
        print(f"   👉 Runtime: {mode_msg} | Selected {len(self.active_test_batch)} images from Test Pool.")

    def get_few_shot_examples(self, k=3):
        """
        Retrieves k random examples from the ACTIVE TRAINING POOL.
        """
        if not self.active_train_pool:
            return []
            
        selected = random.sample(self.active_train_pool, min(k, len(self.active_train_pool)))
        examples = []
        
        for img_name in selected:
            json_name = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(self.desc_dir, json_name)
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    examples.append({
                        "context": data.get('context'),
                        "response": {
                            "perception": data.get('perception'),
                            "prediction": data.get('prediction'),
                            "planning": data.get('planning')
                        }
                    })
        return examples

    def get_test_batch(self):
        batch = []
        for img_name in self.active_test_batch:
            image_path = os.path.join(self.images_dir, img_name)
            json_name = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(self.desc_dir, json_name)
            
            gt = {}
            if os.path.exists(json_path):
                with open(json_path, 'r') as f: gt = json.load(f)
            
            batch.append({
                "id": img_name,
                "image_path": image_path,
                "context": gt.get('context', ''),
                "goal": "Drive safely.",
                "ground_truth": gt
            })
        return batch