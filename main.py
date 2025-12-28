from imports import *
from dataloader import *
from train import *
from data_transformer import *
from bias_finder  import *
from data_subsampler import *
import os
import time

def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__=="__main__":

  
  # 启用 WANDB：移除禁用设置
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--working_dir', type=str)
  parser.add_argument('--config', type=str)
  parser.add_argument('--preprocess_only', action='store_true')
  parser.add_argument('--num_train_epochs', type=int)
  parser.add_argument('--learning_rate', type=float)
  parser.add_argument('--train_batch_size', type=int)
  parser.add_argument('--num_samples', type=int)
  parser.add_argument('--seed', type=int)
  parser.add_argument('--model_name_or_path', type=str)
  parser.add_argument('--experiment_id', type=int)
  parser.add_argument('--run_tag', type=str)
  parser.add_argument('--sample_method', type=str, choices=["most-biased", "random", "diverse-biased"], help="Sampling strategy for subsampling")
  parser.add_argument('--enable_soft_gender_equalize', action='store_true')
  parser.add_argument('--soft_equalize_lambda', type=float)
  parser.add_argument('--sym_reg_lambda', type=float)

  config = {
        "name_mask_method":"smart_random_masking",
        "nonname_mask_method":"smart_random_masking",
        "naive_mask_token":"person",
        "seed":701,
        "persistence_dir":"working_dir",
        "save_data":True,
        "experiment_id":1,
        "load_saved_data":False,
        "epochs":1,
        "learning_rate":5e-5,
        "train_batch_size":10,
        "max_length":512,
        "mlm_prob":1.0,
        "model":"bert-base-uncased",
        "dataset_type":"txt",
        "dataset":"/root/data_debias-main/data/train.txt",
        "sample":10,
        "measure_initial_bias":True,
        "sample_method":"most-biased",
          "diverse_top_k":50,
        "enable_soft_gender_equalize":False,
        "soft_equalize_lambda":1.0,
        "sym_reg_lambda":0.1,
        "gender_pairs":[["she","he"],["her","him"],["hers","his"]],

  }
  args = parser.parse_args()

  #config = json.loads(args.config)
  root =  args.working_dir

  # Override config with CLI args when provided
  if args.num_train_epochs:
    config["epochs"] = args.num_train_epochs
  if args.learning_rate:
    config["learning_rate"] = args.learning_rate
  if args.train_batch_size:
    config["train_batch_size"] = args.train_batch_size
  if args.num_samples:
    config["sample"] = args.num_samples
  if args.seed:
    config["seed"] = args.seed
  if args.model_name_or_path:
    config["model"] = args.model_name_or_path
  if args.experiment_id:
    config["experiment_id"] = args.experiment_id
  if args.sample_method:
    config["sample_method"] = args.sample_method
  if args.enable_soft_gender_equalize:
    config["enable_soft_gender_equalize"] = True
  if args.soft_equalize_lambda is not None:
    config["soft_equalize_lambda"] = args.soft_equalize_lambda
  if args.sym_reg_lambda is not None:
    config["sym_reg_lambda"] = args.sym_reg_lambda
  # run_tag 不写入 config，单次运行用于命名

  random_seed(config["seed"])
  
  wandb.login(key="f05c4a6e8e88d1b93998ff5140c0f738225ae7e2")

  unique_suffix = int(time.time())
  name_parts = [
      f"exp{config['experiment_id']}",
      f"-seed{config['seed']}",
      f"-epochs{config['epochs']}",
      f"-bs{config.get('train_batch_size', 10)}",
      f"-sample{config.get('sample')}",
      f"-{config['model']}"
  ]
  if config.get("enable_soft_gender_equalize"):
    name_parts.append("-softprob")
  if args.run_tag:
      name_parts.append(f"-tag{args.run_tag}")
  name_parts.append(f"-t{unique_suffix}")
  run_name = "".join(name_parts)

  # 将统一的 run_name 放入 config，Trainer 使用同名
  config["run_name"] = run_name

  run = wandb.init(
      name = run_name,
      group = f"exp{config['experiment_id']}",
      reinit = True, 
      project = "data_debias", 
      config = config
  )
  
  dataloader = CustomDataset(config, root, run)
  dataset = dataloader.load()
  raw_dataset = copy.deepcopy(dataset)

  # 评估初始偏见（Original）并保留用于后续采样
  original_avg_bias = None
  if "measure_initial_bias" in config and config["measure_initial_bias"]:
    bias_finder_pre = BiasFinder(config, root, run, model=config["model"])
    dataset = bias_finder_pre.modify_data(raw_dataset)
    try:
      original_avg_bias = float(np.mean(dataset["bias"]))
    except Exception:
      original_avg_bias = None
  
  if "sample" in config:
    dataSubsampler = DataSubsampler(config, root, run, config["model"])
    dataset = dataSubsampler.subsample(dataset, config["sample"], config["sample_method"])

  dataTransformer = DataTransformer(config, root, run)
  dataset, new_words = dataTransformer.modify_data(dataset) #dataset, sample and intervention
  #print(dataset[0])
  model_path = None
  if not args.preprocess_only:
    customTrainer = CustomTrainer(config, root, run)
    model_path = customTrainer.train(dataset, new_words)

    # 训练后用微调模型重新评估偏见（Debiased）
    debiased_avg_bias = None
    try:
      bias_finder_post = BiasFinder(config, root, run, model=model_path)
      post_dataset = bias_finder_post.modify_data(raw_dataset)
      debiased_avg_bias = float(np.mean(post_dataset["bias"]))
    except Exception as e:
      print("Post-training bias evaluation failed:", e)

    # 计算降幅并记录到 W&B 与本地 CSV
    reduction_pct = None
    if original_avg_bias is not None and debiased_avg_bias is not None and original_avg_bias > 0:
      reduction_pct = ((original_avg_bias - debiased_avg_bias) / original_avg_bias) * 100.0

    metrics_row = {
      "experiment_id": config["experiment_id"],
      "seed": config["seed"],
      "sample": config.get("sample"),
      "epochs": config.get("epochs"),
      "original_bias": original_avg_bias,
      "debiased_bias": debiased_avg_bias,
      "reduction_pct": reduction_pct,
      "model": config["model"]
    }

    try:
      run.log({
        "original_bias": original_avg_bias,
        "debiased_bias": debiased_avg_bias,
        "reduction_pct": reduction_pct
      })
    except Exception:
      pass

    # 保存本地表格
    metrics_dir = os.path.join(root, config["persistence_dir"], "metrics", str(config["experiment_id"]))
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, f"seed_{config['seed']}.csv")
    try:
      df = pd.DataFrame([metrics_row])
      if os.path.exists(metrics_file):
        old = pd.read_csv(metrics_file)
        df = pd.concat([old, df], ignore_index=True)
      df.to_csv(metrics_file, index=False)
    except Exception as e:
      print("Failed to save metrics CSV:", e)

  run.finish()
  if model_path is not None:
    print(model_path)
