from imports import *

class CustomDataset:

    def __init__(self, config: dict, root: str = ".", wandb=None):
        self.root = root
        self.config = config
        self.wandb = wandb

    def load(self):
        
        config = self.config

        if config["dataset_type"]=="remote":
            dataset = load_dataset(config["dataset"], split="train[:50]")
            dataset = dataset.rename_column("context", "text") #for squad
        
        elif config["dataset_type"]=="txt":    
            # Accept either a single path string or an iterable of paths
            dataset_paths = config["dataset"]
            if isinstance(dataset_paths, str):
                dataset_paths = [dataset_paths]

            text = []
            for path in dataset_paths:
                # Allow relative paths (resolved from root) and absolute paths
                file_path = path if os.path.isabs(path) else os.path.join(self.root, path)
                with open(file_path, "r") as f:
                    text += [x.strip() for x in f.readlines()]

            dataset = Dataset.from_dict({"text": text})

        elif config["dataset_type"]=="csv":
            dataset = load_dataset("csv", data_files=config["dataset"], split="train")
            dataset = dataset.rename_column("context", "text")
        
        return dataset