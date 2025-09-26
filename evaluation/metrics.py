import numpy as np
import torch


def compute_ece(probabilities, labels, n_bins=15):
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = predictions == labels

    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_bounds[i]
        bin_upper = bin_bounds[i + 1]
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def compute_B_std(hypernet, device="cuda"):
    with torch.no_grad():
        Bs = []
        num_of_matrices = 5
        for _ in range(num_of_matrices):
            if not hypernet.use_batches:
                _, B = hypernet(0, device)  # [hidden, r]
            else:
                hypernet.precompute(device)
                _, B = hypernet.use_precomputed(0)
            Bs.append(B.cpu().numpy())
        Bs = np.stack(Bs)  # shape: [num_of_matrices, hidden, r]
        std_per_batch_element = np.std(
            Bs.reshape(num_of_matrices, -1), axis=0
        )  # shape: [5]
        mean_std = np.mean(std_per_batch_element)
        return float(mean_std)


def compute_B_mean(hypernet, device="cuda"):
    with torch.no_grad():
        Bs = []
        num_of_matrices = 5
        for _ in range(num_of_matrices):
            if not hypernet.use_batches:
                _, B = hypernet(0, device)  # [hidden, r]
            else:
                hypernet.precompute(device)
                _, B = hypernet.use_precomputed(0)
            Bs.append(B.cpu().numpy())
        Bs = np.stack(Bs)  # shape: [num_of_matrices, hidden, r]
        mean_per_batch_element = np.mean(
            Bs.reshape(num_of_matrices, -1), axis=0
        )  # shape: [5]
        mean_mean = np.mean(mean_per_batch_element)
        return float(mean_mean)
