import evaluate
from datasets import load_dataset


def get_glue_dataset(dataset_name, tokenizer, truncation=True, max_length=512):
    def tokenize_function(examples):
        # Map GLUE subsets to their corresponding text field names
        single_sentence_tasks = {"cola", "sst2", "sst-2"}
        pair_sentence_tasks_premise_hypothesis = {"mnli"}
        pair_sentence_tasks_sentence12 = {"mrpc", "rte", "wnli", "stsb"}
        pair_sentence_tasks_question12 = {"qqp"}
        pair_sentence_tasks_question_sentence = {"qnli"}

        if dataset_name.lower() in single_sentence_tasks:
            return tokenizer(
                examples["sentence"],
                truncation=truncation,
                padding="max_length",
                max_length=max_length,
            )

        if dataset_name.lower() in pair_sentence_tasks_premise_hypothesis:
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=truncation,
                padding="max_length",
                max_length=max_length,
            )

        if dataset_name.lower() in pair_sentence_tasks_question_sentence:
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=truncation,
                padding="max_length",
                max_length=max_length,
            )

        if dataset_name.lower() in pair_sentence_tasks_question12:
            return tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=truncation,
                padding="max_length",
                max_length=max_length,
            )

        if dataset_name.lower() in pair_sentence_tasks_sentence12:
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=truncation,
                padding="max_length",
                max_length=max_length,
            )

        # Fallback: try common patterns by presence
        if "sentence" in examples:
            return tokenizer(
                examples["sentence"],
                truncation=truncation,
                padding="max_length",
                max_length=max_length,
            )
        if "sentence1" in examples and "sentence2" in examples:
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=truncation,
                padding="max_length",
                max_length=max_length,
            )
        if "question" in examples and "sentence" in examples:
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=truncation,
                padding="max_length",
                max_length=max_length,
            )
        if "question1" in examples and "question2" in examples:
            return tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=truncation,
                padding="max_length",
                max_length=max_length,
            )
        raise ValueError(f"Unsupported GLUE dataset fields for: {dataset_name}")

    dataset = load_dataset("glue", dataset_name)
    metric = evaluate.load("glue", dataset_name)
    
    # Get the number of labels from the dataset
    num_labels = dataset["train"].features["label"].num_classes

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    
    # MNLI has validation_matched and validation_mismatched instead of validation
    # Rename validation_matched to validation for consistency with other tasks
    if dataset_name.lower() == "mnli":
        if "validation_matched" in encoded_dataset:
            encoded_dataset["validation"] = encoded_dataset["validation_matched"]
    
    encoded_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return encoded_dataset, metric, num_labels
