params = {
    # most important, general params
    "glue_dataset_name": "cola",
    "model_name": "roberta-base",
    "use_hypernet": False,
    # which layeres to freeze (not necessary when using peft lora, since it automatically freezes them)
    "layers_to_freeze": [],
    # peft LoRA params (https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig)
    "use_peft": True,
    "lora_r": 1,
    "lora_alpha": 16,
    "target_modules": ["query", "value"],
    "layers_to_transform": list(range(12)),
    "layers_pattern": "encoder.layer",
    "forward_pass_reps": 1,
    # transformers trainer args (https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/trainer#transformers.TrainingArguments)
    "output_dir": f"./pretrained_models/baseline",
    "eval_strategy": "epoch",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 1000000000,
    "logging_strategy": "epoch",
    "logging_steps": 50,
    "learning_rate": 4e-4,
    "weight_decay": 0.1,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 80,
    "metric_for_best_model": "matthews_correlation",
    "warmup_ratio": 0.06,
    "lr_scheduler_type": "linear",
    "optim": "adamw_torch",
    "disable_tqdm": True,
    # filenames are being generated based on this filename and timestep to avoid overwriting previous results
    "results_dir": "./results/output_dir",
    "save_model_at_the_end": False,  # in most cases we want to save the model at the end
    "num_runs": 3,  # we will train this many times with this config, but seeds will be different
    "seed": 11,
}
