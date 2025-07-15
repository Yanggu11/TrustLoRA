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
        for _ in range(5):
            A = torch.randn((hypernet.lora_r, hypernet.input_dim)).to(device)
            B = hypernet(A)  # [hidden, r]
            Bs.append(B.cpu().numpy())
        Bs = np.stack(Bs)  # shape: [5, hidden, r]
        std_per_element = np.std(Bs, axis=0)
        mean_std = np.mean(std_per_element)
        return float(mean_std)

def compute_B_mean(hypernet, device="cuda"):
    with torch.no_grad():
        Bs = []
        for _ in range(5):
            A = torch.randn((hypernet.lora_r, hypernet.input_dim)).to(device)
            B = hypernet(A)  # [hidden, r]
            Bs.append(B.cpu().numpy())
        Bs = np.stack(Bs)  # shape: [5, hidden, r]
        mean_per_element = np.mean(Bs, axis=0)
        avg_mean = np.mean(mean_per_element)
        return float(avg_mean)
