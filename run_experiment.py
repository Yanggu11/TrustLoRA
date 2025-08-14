import time

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from data_loading.get_datasets import get_glue_dataset
from evaluation.lr_scheduler_callback import ReduceLROnPlateauCallback
from evaluation.metrics import compute_B_mean, compute_B_std, compute_ece
from evaluation.metrics_trainer_callback import SaveMetricsCallback
from models.get_roberta import get_baseline_roberta, get_hypernet_on_last_layer_roberta

import argparse
import importlib.util
import os

def run_experiment(params, i, device="cpu"):
    print(f"=== Run {i} ==============")

    experiment_id = str(int(time.time()))

    if params["use_hypernet"]:
        model, tokenizer, hypernet = get_hypernet_on_last_layer_roberta(
            model_name=params["model_name"],
            lora_r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            hypernet_hidden_dim=params["hypernet_hidden_dim"],
            hypernet_embeddings_dim=params["hypernet_embeddings_dim"],
            use_on_value_matrix=params["hypernet_use_on_value_matrix"],
            hypernet_with_embedding_input_only=params[
                "hypernet_with_embedding_input_only"
            ],
            use_fixed_A=params["hypernet_use_fixed_A"],
        )
    else:
        model, tokenizer = get_baseline_roberta(
            model_name=params["model_name"],
            lora_r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
        )
    encoded_dataset, metric = get_glue_dataset(
        params["glue_dataset_name"], tokenizer, truncation=True, max_length=512
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

        predictions = np.argmax(probs, axis=-1)
        results = metric.compute(predictions=predictions, references=labels)

        results["ece"] = compute_ece(probs, labels)
        if params["use_hypernet"]:
            results["hyper_B_std"] = compute_B_std(hypernet, device=device)
            results["hyper_B_mean"] = compute_B_mean(hypernet, device=device)

        return results

    training_args = TrainingArguments(
        output_dir=f"{params['output_dir']}_{params['glue_dataset_name']}_{experiment_id}",
        eval_strategy=params["eval_strategy"],
        eval_steps=params["eval_steps"] if params["eval_strategy"] == "steps" else None,
        save_strategy=params["save_strategy"],
        save_steps=params["save_steps"] if params["save_strategy"] == "steps" else None,
        logging_strategy=params["logging_strategy"],
        logging_steps=(
            params["logging_steps"] if params["logging_strategy"] == "steps" else None
        ),
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        per_device_eval_batch_size=params["per_device_eval_batch_size"],
        num_train_epochs=params["num_train_epochs"],
        metric_for_best_model=params["metric_for_best_model"],
        warmup_ratio=params["warmup_ratio"],
        lr_scheduler_type=params["lr_scheduler_type"],
        optim=params["optim"],
        weight_decay=params["weight_decay"],
        disable_tqdm=params["disable_tqdm"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            SaveMetricsCallback(
                params["results_dir"],
                f"{params['results_filename']}_{params['glue_dataset_name']}_{experiment_id}.csv",
            )
        ],
    )

    trainer.evaluate()
    trainer.train()

def load_params_from_file(file_path):
    spec = importlib.util.spec_from_file_location("params_module", file_path)
    params_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params_module)
    return params_module.params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to params .py file"
    )
    args = parser.parse_args()

    # Load params
    if not os.path.exists(args.params):
        raise FileNotFoundError(f"Params file {args.params} does not exist.")

    params = load_params_from_file(args.params)
    print("Loaded params:", params)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Hypernet on: {params['glue_dataset_name']}")

    for i in range(params["num_runs"]):
        run_experiment(params, i, device=device)
    
