import os
import json
import random

class SplitFolderDataset:
    def __init__(self, root_dir, split_ratio=0.8, seed=42):
        self.root_dir = os.path.abspath(root_dir)
        self.images_dir = os.path.join(self.root_dir, "images")
        self.desc_dir = os.path.join(self.root_dir, "descriptions")
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images folder not found at {self.images_dir}")

        all_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))])
        
        random.seed(seed)
        random.shuffle(all_files)
        
        split_idx = int(len(all_files) * split_ratio)
        self.train_files = all_files[:split_idx]
        self.test_files = all_files[split_idx:]
        
        print(f"[Dataset] Split Loaded: {len(self.train_files)} Training | {len(self.test_files)} Testing")

    def get_train_data(self, limit=None):
        """
        Fetches training examples. 
        If limit is set, returns that many examples (for the 2:1 ratio).
        """
        training_examples = []
        
        # Use the limit if provided, otherwise use all
        target_files = self.train_files[:limit] if limit else self.train_files
        
        for img_name in target_files:
            data = self._load_single_item(img_name)
            if data:
                training_examples.append(data[3])
                
        return training_examples

    def get_test_data(self):
        return [self._load_single_item(f) for f in self.test_files]

    def _load_single_item(self, img_name):
        image_path = os.path.join(self.images_dir, img_name)
        json_name = os.path.splitext(img_name)[0] + ".json"
        json_path = os.path.join(self.desc_dir, json_name)
        
        ground_truth = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                ground_truth = json.load(f)
        
        context = ground_truth.get("context", "Urban driving scenario")
        goal = ground_truth.get("goal", "Drive safely")
        
        return image_path, context, goal, ground_truth