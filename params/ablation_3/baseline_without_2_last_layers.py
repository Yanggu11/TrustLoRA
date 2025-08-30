params = {
    "glue_dataset_name": "cola",
    "model_name": "roberta-base",
    "use_hypernet": False,

    "lora_r": 8,
    "lora_alpha": 16,

    "target_modules": ["query", "value"],
    "layers_to_transform": list(range(10)),
    "layers_pattern": "encoder.layer",


    "output_dir": f"./outputs/LoRA_baseline",
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
    "num_train_epochs": 20,
    "metric_for_best_model": "matthews_correlation",
    "warmup_ratio": 0.06,
    "lr_scheduler_type": "linear",
    "optim": "adamw_torch",
    "disable_tqdm": True,

    "results_dir": "./results/baseline_without_last_2",

    "num_runs": 2,
    "seed": 11
}
