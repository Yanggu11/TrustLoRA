import time

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from data_loading.get_datasets import get_glue_dataset
from utils.metrics import compute_B_mean, compute_B_std, compute_ece
from utils.metrics_trainer_callback import SaveMetricsCallback
from models.get_roberta import get_hypernet_on_last_layer_roberta

device = "cuda" if torch.cuda.is_available() else "cpu"

glue_dataset_name = "cola"
model_name = "roberta-base"
lora_r = 4
lora_alpha = 16
hypernet_hidden_dim = 16
hypernet_embeddings_dim = 64

print(f"Hypernet on: {glue_dataset_name}")

for i in range(1):
    print(f"=== Run {i} ==============")

    model, tokenizer, hypernet = get_hypernet_on_last_layer_roberta(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        hypernet_hidden_dim=hypernet_hidden_dim,
        hypernet_embeddings_dim=hypernet_embeddings_dim,
        use_on_value_matrix=True,
        hypernet_with_embedding_input=True,
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
        results["hyper_B_std"] = compute_B_std(hypernet, device=device)
        results["hyper_B_mean"] = compute_B_mean(hypernet, device=device)

        return results

    training_args = TrainingArguments(
        output_dir=f"./outputs/hypernet_{glue_dataset_name}",
        eval_strategy="epoch",
        # eval_steps=40,
        save_strategy="steps",
        save_steps=1000000000,
        learning_rate=4e-4,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=32,
        num_train_epochs=20,  # 80
        logging_dir=f"./logs/hypernet_{glue_dataset_name}",
        logging_strategy="epoch",
        metric_for_best_model="matthews_correlation",
        dataloader_num_workers=4,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        weight_decay=0.1,
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
                f"g_emb_vs_hid_dim_4_{glue_dataset_name}_{str(int(time.time()))}.csv",
            )
        ],
    )

    trainer.train()
