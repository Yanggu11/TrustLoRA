params = {
    # most important, general params
    "glue_dataset_name": "stsb",
    "model_name": "roberta-large",
    "use_hypernet": False,
    # which layeres to freeze (not necessary when using peft lora, since it automatically freezes them)
    "layers_to_freeze": [],
    # peft LoRA params (https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig)
    "use_peft": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "target_modules": ["query", "value"],
    "layers_to_transform": list(range(24)),  # all layers
    "layers_pattern": "encoder.layer",
    "forward_pass_reps": 1,
    # transformers trainer args (https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/trainer#transformers.TrainingArguments)
    "output_dir": f"./pretrained_models/cola_baseline",
    "eval_strategy": "epoch",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 10000000000,
    "logging_strategy": "epoch",
    "logging_steps": 50,
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 30,
    "metric_for_best_model": "matthews_correlation",
    "warmup_ratio": 0.06,
    "lr_scheduler_type": "linear",
    "optim": "adamw_torch",
    "disable_tqdm": True,
    # filenames are being generated based on this filename and timestep to avoid overwriting previous results
    "results_dir": "./results/cola",
    "num_runs": 1,  # we will train this many times with this config, but seeds will be different
    "seed": 11,
}
