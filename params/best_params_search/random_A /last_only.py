params = {
    # most important, general params
    "glue_dataset_name": "cola",
    "model_name": "roberta-base",
    "use_hypernet": True,
    # which layeres to freeze (not necessary when using peft lora, since it automatically freezes them)
    "layers_to_freeze": [],
    # peft LoRA params (https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig)
    "use_peft": True,
    "lora_r": 8,
    "lora_alpha": 8,
    "target_modules": ["query", "value"],
    "layers_to_transform": [11],
    "layers_pattern": "encoder.layer",
    # hypernet params
    "hypernet_use_embedding": True,  # if False use one-hot encoding
    "hypernet_use_transformer": False, # !
    "hypernet_transformer_nhead": 8,
    "hypernet_transformer_num_layers": 2,
    "hypernet_noise_type_A": "replace",  # "replace", "add", "multiply"
    "hypernet_noise_type_B": "replace",  # "replace", "add", "multiply"
    "hypernet_reduce_noise_alpha": False,
    "hypernet_noise_alpha": 1,
    "hypernet_use_batches": True, # !
    "hypernet_hidden_dim": [2048, 2048, 2048],  # if large_model is True use 3 layers, otherwise use single hidden layer
    "hypernet_embeddings_dim": 128,
    "layers_to_use_hypernet": [11],
    "hypernet_use_on_value_matrix": True,  # by default we apply lora only on query matrix if this is set to False
    "hypernet_with_embedding_input_only": True,  # if False we concat matrix A and embedding as input to hypernet
    "hypernet_large_model": True,  # if True hypernet has 4 layers, 2 layers otherwise
    "hypernet_A_matrix": "fixed",  # ["random", "fixed", "generated"]
    # in most cases this param is 1, it says how many time in a row we should run forward pass on single batch
    "forward_pass_reps": 1,
    # transformers trainer args (https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/trainer#transformers.TrainingArguments)
    "output_dir": f"./pretrained_models/hy_cola",
    "eval_strategy": "epoch",
    "eval_steps": 5,
    "save_strategy": "steps",
    "save_steps": 1000000000,
    "logging_strategy": "epoch",
    "logging_steps": 50,
    "learning_rate": 4e-5,
    "weight_decay": 0.1,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 20,
    "metric_for_best_model": "matthews_correlation",
    "warmup_ratio": 0.06,
    "lr_scheduler_type": "linear",
    "optim": "adamw_torch",
    "disable_tqdm": False,
    # filenames are being generated based on this filename and timestep to avoid overwriting previous results
    "results_dir": "./results/hy_cola",
    "num_runs": 3,  # we will train this many times with this config, but seeds will be different
    "seed": 11,
}
