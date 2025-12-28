from imports import *

class CustomDataCollator(DataCollatorForLanguageModeling):

  def __init__(self, tokenizer, mlm = True, mlm_probability=None, pad_to_multiple_of= None, tf_experimental_compile = False, return_tensors = "pt"):
    super().__init__(tokenizer, mlm = True, mlm_probability = mlm_probability, pad_to_multiple_of= None, tf_experimental_compile = False, return_tensors = "pt")
  
  def torch_mask_tokens(self, inputs, special_tokens_mask):
    # Ensure integer token ids for indexing ops
    inputs = inputs.to(torch.long)
    labels = inputs.clone()

    probability_matrix = torch.full(labels.shape, self.mlm_probability, device=inputs.device)
    if special_tokens_mask is None:
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    replace_tokens = CustomTrainer.replace_tokens

    # Only predict tokens we explicitly want to neutralize; if none are present, fall back to standard masking
    replace_tensor = torch.tensor(replace_tokens, device=inputs.device, dtype=torch.long)
    target_mask = masked_indices & torch.isin(inputs, replace_tensor)
    if not target_mask.any():
        target_mask = masked_indices

    labels[~target_mask] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool() & target_mask
    inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool() & target_mask & ~indices_replaced
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=inputs.device)

    inputs[indices_random] = random_words[indices_random]

    for l in labels:
        words = self.tokenizer.decode(l[l != -100])
        if len(words) > 0:
            CustomTrainer.mlm_words += words.split(" ")

    return inputs, labels

  def torch_call(self, features):
      """Ensure token_type_ids stay in integer dtype for embedding lookup."""
      batch = super().torch_call(features)
      # Force all id-like tensors to long to avoid float token_type_ids on GPU
      for key in ["input_ids", "token_type_ids", "attention_mask", "labels"]:
          if key in batch:
              batch[key] = batch[key].long()
      return batch


class SoftEqualizeTrainer(Trainer):
    """Trainer with optional soft gender equalization loss to reduce over-correction."""

    def __init__(self, *args, gender_pairs_ids=None, soft_lambda: float = 1.0, sym_lambda: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gender_pairs_ids = gender_pairs_ids or []
        self.soft_lambda = soft_lambda
        self.sym_lambda = sym_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        # Base MLM loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Early exit if no extra constraints
        if not self.gender_pairs_ids:
            return (loss, outputs) if return_outputs else loss

        log_probs = F.log_softmax(logits, dim=-1)
        soft_losses = []
        sym_losses = []

        for pair in self.gender_pairs_ids:
            if len(pair) != 2:
                continue
            a_id, b_id = pair
            pair_tensor = torch.tensor(pair, device=labels.device)
            mask = torch.isin(labels, pair_tensor)
            if not mask.any():
                continue

            p_a = torch.exp(log_probs[..., a_id])
            p_b = torch.exp(log_probs[..., b_id])
            p_avg = 0.5 * (p_a + p_b)
            soft_term = -torch.log(torch.clamp(p_avg, min=1e-12))
            soft_losses.append(soft_term[mask].mean())

            if self.sym_lambda > 0:
                diff = log_probs[..., a_id] - log_probs[..., b_id]
                sym_losses.append((diff[mask] ** 2).mean())

        if soft_losses:
            loss = loss + self.soft_lambda * torch.stack(soft_losses).mean()
        if sym_losses:
            loss = loss + self.sym_lambda * torch.stack(sym_losses).mean()

        return (loss, outputs) if return_outputs else loss

class CustomTrainer:

    mlm_words = []
    replace_tokens = []

    def __init__(self, config: dict, root: str = ".", wandb=None):
        self.config = config
        self.root = root
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.wandb = wandb

    def _group_texts(self, examples):
        # Concatenate all texts.
        max_length = self.max_length
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    def train(self, dataset, words):
        
        dataset = dataset.remove_columns("bias")

        d = dataset.train_test_split(test_size=0.1)
        def dataset_to_text(dataset, output_filename="data.txt"):
            """Utility function to save dataset text to disk,
            useful for using the texts to train the tokenizer 
            (as the tokenizer accepts files)"""
            with open(os.path.join(tempfile.gettempdir(), output_filename), "w") as f:
                for t in dataset["text"]:
                    print(t, file=f)

        dataset_to_text(d["train"], "train.txt")
        dataset_to_text(d["test"], "test.txt")

        # special_tokens = [
        #     "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
        # ]
        
        # files = ["train.txt"]
        vocab_size = 30_522
        max_length = self.config["max_length"]
        self.max_length = max_length
        truncate_longer_samples = False
        
        # # initialize the WordPiece tokenizer
        #tokenizer = BertWordPieceTokenizer.from_pretrained("bert-base-uncased")
        # # train the tokenizer
        #tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
        # # enable truncation up to the maximum 512 tokens
        #tokenizer.enable_truncation(max_length=max_length)

        model_path = "pretrained"
        # Make sure the temp directory exists without raising if already there
        os.makedirs(os.path.join(tempfile.gettempdir(), model_path), exist_ok=True)

        # # save the tokenizer  
        #tokenizer.save_model(model_path)

        # # dumping some of the tokenizer config to config file, 
        # # including special tokens, whether to lower case and the maximum sequence length
        # with open(os.path.join(model_path, "config.json"), "w") as f:
        #   tokenizer_cfg = {
        #       "do_lower_case": True,
        #       "unk_token": "[UNK]",
        #       "prompt_token": "[PMT]",
        #       "sep_token": "[SEP]",
        #       "pad_token": "[PAD]",
        #       "cls_token": "[CLS]",
        #       "mask_token": "[MASK]",
        #       "model_max_length": max_length,
        #       "max_len": max_length,
        #   }
        #   json.dump(tokenizer_cfg, f)

        # when the tokenizer is trained and configured, load it as BertTokenizerFast
        
        tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        #special_tokens_dict = {'additional_special_tokens': ['[START]', '[STOP]']}
        
        #num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        # check if the tokens are already in the vocabulary
        new_tokens = set(words) - set(tokenizer.vocab.keys())

        # add the tokens to the tokenizer vocabulary
        tokenizer.add_tokens(list(new_tokens))

        # Build replace token ids: new tokens + optional gender tokens for soft equalization
        encoded_new_tokens = tokenizer.encode(" ".join(list(new_tokens)), add_special_tokens=False)

        gender_pairs_tokens = self.config.get(
            "gender_pairs", [["she", "he"], ["her", "him"], ["hers", "his"]]
        )
        gender_pairs_ids = []
        for pair in gender_pairs_tokens:
            ids = [tokenizer.convert_tokens_to_ids(tok) for tok in pair]
            if len(ids) == 2 and all(tid not in tokenizer.all_special_ids for tid in ids):
                gender_pairs_ids.append(ids)

        gender_token_ids = list({tid for pair in gender_pairs_ids for tid in pair})

        replace_ids = encoded_new_tokens + gender_token_ids
        replace_ids = [tid for tid in replace_ids if tid not in tokenizer.all_special_ids]
        CustomTrainer.replace_tokens = list(sorted(set(replace_ids)))

        def encode_with_truncation(examples):
            """Mapping function to tokenize the sentences passed with truncation"""
            return tokenizer(examples["text"], truncation=True, padding="max_length",
                            max_length=max_length, return_special_tokens_mask=True,
                            return_token_type_ids=False)

        def encode_without_truncation(examples):
            """Mapping function to tokenize the sentences passed without truncation"""
            return tokenizer(examples["text"], return_special_tokens_mask=True,
                            return_token_type_ids=False)

        encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation
        train_dataset = d["train"].map(encode, batched=True)
        test_dataset = d["test"].map(encode, batched=True)

        if truncate_longer_samples:
            train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
            test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        else:
            test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
            train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
       

        if not truncate_longer_samples:
            train_dataset = train_dataset.map(self._group_texts, batched=True,
                                            desc=f"Grouping texts in chunks of {max_length}")
            test_dataset = test_dataset.map(self._group_texts, batched=True,
                                            desc=f"Grouping texts in chunks of {max_length}")
            # convert them from lists to torch tensors
            train_dataset.set_format("torch")
            test_dataset.set_format("torch")

        # initialize the model with the config
        model_config = AutoConfig.from_pretrained(self.config["model"])#(self.config["model"], vocab_size=vocab_size, max_position_embeddings=max_length)
        model = AutoModelForMaskedLM.from_pretrained(self.config["model"], config=model_config)
        model.resize_token_embeddings(len(tokenizer))
        
        # for name, param in model.named_parameters():
        #    print(name, param.requires_grad)

        # sys.exit(0)

        data_collator = CustomDataCollator(
            tokenizer=tokenizer, mlm=True, mlm_probability=self.config["mlm_prob"]
        )
        
        
        # Compose or reuse a descriptive W&B run name to distinguish seeds/runs
        default_run_name = (
            f"exp{self.config['experiment_id']}-"
            f"seed{self.config['seed']}-"
            f"epochs{self.config['epochs']}-"
            f"bs{self.config.get('train_batch_size', 10)}-"
            f"sample{self.config.get('sample')}-"
            f"{self.config['model']}"
        )
        run_name = self.config.get("run_name", default_run_name)

        training_args = TrainingArguments(
            output_dir=tempfile.gettempdir(),          # output directory to where save model checkpoint
            eval_strategy="epoch",                     # evaluate at the end of each epoch
            overwrite_output_dir=True,      
            num_train_epochs=self.config["epochs"],    # number of training epochs
            per_device_train_batch_size=self.config.get("train_batch_size", 10),
            gradient_accumulation_steps=8,  # accumulate gradients before weight update
            per_device_eval_batch_size=max(1, self.config.get("train_batch_size", 10) * 6),
            learning_rate=self.config.get("learning_rate", 5e-5),
            logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
            save_steps=1000,
            load_best_model_at_end=True,    # load the best model (lowest loss) at the end of training
            save_total_limit=1,             # keep only 1 checkpoint to save disk space
            save_strategy = "epoch",
            seed=self.config["seed"],
            report_to="wandb",
            run_name=run_name

        )

        # initialize the trainer (optionally with soft gender equalization loss)
        trainer_cls = Trainer
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": test_dataset,
            "data_collator": data_collator,
        }

        if self.config.get("enable_soft_gender_equalize", False):
            trainer_cls = SoftEqualizeTrainer
            trainer_kwargs.update({
                "gender_pairs_ids": gender_pairs_ids,
                "soft_lambda": self.config.get("soft_equalize_lambda", 1.0),
                "sym_lambda": self.config.get("sym_reg_lambda", 0.1),
            })

        trainer = trainer_cls(**trainer_kwargs)

        # train the model
        gc.collect()
        torch.cuda.empty_cache()
        trainer.train()
        
        df = pd.DataFrame.from_records(list(dict(Counter(CustomTrainer.mlm_words)).items()), columns=['word','count']).sort_values(by="count")
        df = wandb.Table(dataframe=df)
        self.wandb.log({"mlm words":df})

        self.logger.debug("MLM words: {}".format(CustomTrainer.mlm_words))
        model_path = os.path.join(self.root, self.config["persistence_dir"], "models", str(self.config["experiment_id"]), self.config["model"])
        Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)

        return model_path