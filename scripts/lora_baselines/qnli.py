import time

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from data_loading.get_datasets import get_glue_dataset
from evaluation.metrics import compute_ece
from evaluation.metrics_trainer_callback import SaveMetricsCallback
from models.get_roberta import get_baseline_roberta

device = "cuda" if torch.cuda.is_available() else "cpu"

glue_dataset_name = "qnli"
model_name = "roberta-base"
lora_r = 8
lora_alpha = 16

print(f"LoRA Baseline on: {glue_dataset_name}")

for i in range(2):
    print(f"=== Run {i} ==============")

    model, tokenizer = get_baseline_roberta(
        model_name=model_name, lora_r=lora_r, lora_alpha=lora_alpha
    )
    encoded_dataset, metric = get_glue_dataset(
        glue_dataset_name, tokenizer, truncation=True, max_length=512
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

        predictions = np.argmax(probs, axis=-1)
        results = metric.compute(predictions=predictions, references=labels)

        results["ece"] = compute_ece(probs, labels)
        results["logits_shape_last"] = logits.shape[-1]

        return results

    training_args = TrainingArguments(
        output_dir=f"./outputs/LoRA_baseline_{glue_dataset_name}",
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=1000000000,
        learning_rate=4e-4,
        weight_decay=0.1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=32,
        num_train_epochs=25,
        metric_for_best_model="accuracy",
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        disable_tqdm=True,
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
                f"./results",
                f"LoRA_baseline_{glue_dataset_name}_{str(int(time.time()))}.csv",
            )
        ],
    )

    trainer.train()
