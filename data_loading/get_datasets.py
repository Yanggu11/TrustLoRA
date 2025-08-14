import evaluate
from datasets import load_dataset


def get_glue_dataset(dataset_name, tokenizer, truncation=True, max_length=512):
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=truncation,
            padding="max_length",
            max_length=max_length,
        )

    dataset = load_dataset("glue", dataset_name)
    metric = evaluate.load("glue", dataset_name)

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return encoded_dataset, metric
