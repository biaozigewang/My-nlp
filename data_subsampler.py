from typing import List

from imports import *
from sklearn.cluster import KMeans
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

class DataSubsampler:

  def __init__(self, config: dict, root: str = ".", wandb=None, model="bert-base-uncased"):

    self.config = config
    self.root = root

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    self.logger = logging.getLogger(__name__)

    self.tokenizer = AutoTokenizer.from_pretrained(model)
    self.model = AutoModel.from_pretrained(model, output_hidden_states=True)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model = self.model.to(self.device)
    self.model.eval()

  def _embed_texts(self, texts: List[str]) -> np.ndarray:
    """Return CLS embeddings for a batch of texts using the configured model."""
    enc = self.tokenizer(
        texts,
        max_length=self.config.get("max_length", 512),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(self.device) for k, v in enc.items()}
    with torch.no_grad():
      outputs = self.model(**enc)
    cls = outputs.last_hidden_state[:, 0, :]
    return cls.cpu().numpy()

  def _diverse_biased(self, dataset: Dataset, samples: int) -> Dataset:
    if "bias" not in dataset.column_names:
      raise ValueError("Dataset must contain a 'bias' column for diverse-biased sampling.")

    top_k = int(self.config.get("diverse_top_k", 50))
    sorted_ds = dataset.sort("bias", reverse=True)
    top_k = min(top_k, len(sorted_ds))
    top_subset = sorted_ds.select(range(top_k))

    k = min(samples, len(top_subset))
    texts = top_subset["text"]
    embeddings = self._embed_texts(texts)

    kmeans = KMeans(n_clusters=k, random_state=self.config["seed"])
    labels = kmeans.fit_predict(embeddings)

    selected_indices = []
    # pick the highest-bias point per cluster to keep high bias while diversifying semantics
    for cluster_id in range(k):
      cluster_items = [idx for idx, lbl in enumerate(labels) if lbl == cluster_id]
      if not cluster_items:
        continue
      best_idx = max(cluster_items, key=lambda idx: top_subset[idx]["bias"])
      selected_indices.append(best_idx)

    # ensure we have exactly `samples` rows by filling with remaining highest-bias entries
    if len(selected_indices) < samples:
      remaining = [idx for idx in range(len(top_subset)) if idx not in selected_indices]
      remaining_sorted = sorted(remaining, key=lambda idx: top_subset[idx]["bias"], reverse=True)
      needed = samples - len(selected_indices)
      selected_indices.extend(remaining_sorted[:needed])

    # de-duplicate and keep order by bias priority
    seen = set()
    ordered = []
    for idx in selected_indices:
      if idx not in seen:
        ordered.append(idx)
        seen.add(idx)

    result = top_subset.select(ordered)

    # persist top-k and selected for inspection
    try:
      exp_id = str(self.config.get("experiment_id", "unknown"))
      stats_dir = os.path.join(self.root, self.config.get("persistence_dir", "working_dir"), "intervention_stats", exp_id)
      Path(stats_dir).mkdir(parents=True, exist_ok=True)

      pd.DataFrame(top_subset).to_csv(os.path.join(stats_dir, "top{}_diverse_biased.csv".format(top_k)), index=False)
      pd.DataFrame(result).to_csv(os.path.join(stats_dir, "top{}_diverse_biased_selected.csv".format(samples)), index=False)
    except Exception as e:
      # best-effort save; log and continue
      self.logger.warning("Failed to save diverse-biased selections: %s", e)

    return result

  def subsample(self, dataset, samples, method="random"):

    methods = ["random", "most-biased", "diverse-biased"]
    if method not in methods:
      raise Exception("Method can only be one of: {}".format(",".join(methods)))
    
    if type(samples) is str:
      samples = int(len(dataset)*(int(samples[:-1])/100))
    
    samples = min(samples, len(dataset))
    self.logger.debug("Using {} out of {} rows in the dataset".format(samples, len(dataset)))
    
    if method == "random":
      dataset = dataset.shuffle(seed=self.config["seed"]).select(range(samples))
    elif method == "most-biased":
      dataset = dataset.sort('bias', reverse=True)
      dataset = dataset.select(range(samples))
    elif method == "diverse-biased":
      dataset = self._diverse_biased(dataset, samples)

    return dataset

# if __name__=="__main__":

#   experiments = [
#       {
#         "name_mask_method":"smart_random_masking",
#         "nonname_mask_method":"smart_random_masking",
#         "naive_mask_token":"person",
#         "seed":701,
#         "persistence_dir":"src/logs",
#         "save_data":True,
#         "experiment_id":1,
#         "load_saved_data":False,
#         "max_length":512
#       }
#     ]
    

#   for config in experiments:
#     print(config)
#     dataTransformer = DataSubsampler(config, "/Users/Himanshu/Developer/")
#     dataset = load_dataset("csv", data_files="ssd.csv", split="train")
#     # dataset = dataset.rename_column("context", "text")
#     dataset = dataset.shuffle(seed=701).select(range(20))
#     dataset = dataTransformer.subsample(dataset, 10)
#     print(dataset)
  
