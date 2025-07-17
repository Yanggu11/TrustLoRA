import os

from transformers import TrainerCallback


class SaveMetricsCallback(TrainerCallback):
    def __init__(self, directory, filepath):
        self.directory = directory
        self.filepath = filepath
        
    def _init_file(self, list_of_metrics):
        os.makedirs(self.directory, exist_ok=True)
        if not os.path.exists(os.path.join(self.directory, self.filepath)):
            with open(os.path.join(self.directory, self.filepath), "w") as f:
                f.write(",".join(list_of_metrics) + "\n")  # write header

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self._init_file(list(metrics.keys()))

        file_row = ""
        for key in metrics.keys():
            file_row += str(metrics[key]) + ","
        file_row = file_row[:-1]
        file_row += "\n"

        with open(os.path.join(self.directory, self.filepath), "a") as f:
            f.write(file_row)