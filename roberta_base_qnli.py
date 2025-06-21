from datasets import load_dataset
import numpy as np
import evaluate
from peft import get_peft_model, LoraConfig, TaskType
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

for it in range(3):
    print(f"====== Run {it} ===============")
    
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    base_model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)  # QNLI is binary classification
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=8
    )
    model = get_peft_model(base_model, peft_config)

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load QNLI dataset
    dataset = load_dataset("glue", "qnli")

    # Tokenize sentence pairs
    def tokenize_function(examples):
        # QNLI has 'question' and 'sentence' columns for the pair
        return tokenizer(examples["question"], examples["sentence"], truncation=True, padding="max_length", max_length=128)

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    metric = evaluate.load("glue", "qnli")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./outputs/roberta_base_qnli",
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=1000000,
        learning_rate=4e-4,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=32,
        num_train_epochs=25,
        logging_dir="./logs/roberta_base_qnli",
        metric_for_best_model="accuracy",  # QNLI uses accuracy metric primarily
        dataloader_num_workers=4,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
