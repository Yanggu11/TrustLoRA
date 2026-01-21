import numpy as np


def ece(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate the Expected Calibration Error (ECE) for probabilistic predictions.

    ECE measures the difference between predicted confidence and actual accuracy across
    confidence bins. A lower ECE indicates better calibration, where predicted probabilities
    more accurately reflect the true likelihood of correctness.

    Args:
        probabilities: Array of shape (n_samples, n_classes) containing predicted probabilities
                      for each class. Each row should sum to 1.0.
        labels: Array of shape (n_samples,) containing true class labels as integers.
        n_bins: Number of bins to partition confidence scores into. Default is 10.

    Returns:
        The Expected Calibration Error as a float in the range [0, 1].
        Lower values indicate better calibration.

    Example:
        >>> probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
        >>> labels = np.array([0, 0, 1])
        >>> ece_score = ece(probs, labels, n_bins=10)
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = predictions == labels

    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece_value = 0.0

    for i in range(n_bins):
        bin_lower = bin_bounds[i]
        bin_upper = bin_bounds[i + 1]

        if i == 0:
            in_bin_mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin_mask = (confidences > bin_lower) & (confidences <= bin_upper)

        proportion_in_bin = np.mean(in_bin_mask)

        if proportion_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin_mask])
            avg_confidence_in_bin = np.mean(confidences[in_bin_mask])
            ece_value += (
                np.abs(avg_confidence_in_bin - accuracy_in_bin) * proportion_in_bin
            )

    return ece_value


def classwise_ece(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """
    Calculate the Classwise Expected Calibration Error (CECE).

    CECE is a macro-averaged ECE that extends bin-based ECE to measure calibration
    across all possible classes. Predictions are binned separately for each class,
    and the calibration error is computed at the level of individual class-bins and
    then averaged.

    Args:
        probabilities: Array of shape (n_samples, n_classes) containing predicted probabilities
                      for each class. Each row should sum to 1.0.
        labels: Array of shape (n_samples,) containing true class labels as integers.
        n_bins: Number of bins to partition confidence scores into. Default is 10.

    Returns:
        The Classwise Expected Calibration Error as a float in the range [0, 1].
        Lower values indicate better calibration.

    Example:
        >>> probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
        >>> labels = np.array([0, 0, 1])
        >>> cece_score = classwise_ece(probs, labels, n_bins=10)
    """
    n_samples, n_classes = probabilities.shape
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    cece_value = 0.0

    for c in range(n_classes):

        class_confidences = probabilities[:, c]
        binary_labels = (labels == c).astype(int)

        for i in range(n_bins):
            bin_lower = bin_bounds[i]
            bin_upper = bin_bounds[i + 1]

            if i == 0:
                in_bin_mask = (class_confidences >= bin_lower) & (
                    class_confidences <= bin_upper
                )
            else:
                in_bin_mask = (class_confidences > bin_lower) & (
                    class_confidences <= bin_upper
                )

            n_in_bin = np.sum(in_bin_mask)

            if n_in_bin > 0:
                accuracy_in_bin = np.mean(binary_labels[in_bin_mask])
                avg_confidence_in_bin = np.mean(class_confidences[in_bin_mask])
                cece_value += (n_in_bin / (n_samples * n_classes)) * np.abs(
                    accuracy_in_bin - avg_confidence_in_bin
                )

    return cece_value


def mce(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate the Maximum Calibration Error (MCE) for probabilistic predictions.

    MCE focuses on identifying the largest discrepancy between predicted probabilities
    and empirical accuracies within any individual bin. It measures the worst-case
    calibration error across all confidence bins.

    This metric is particularly important in high-risk applications where reliable
    confidence measures are absolutely necessary. While MCE highlights the most
    substantial calibration error observed within the prediction space, it may not
    offer as comprehensive an assessment of overall calibration performance as ECE does.

    Args:
        probabilities: Array of shape (n_samples, n_classes) containing predicted probabilities
                      for each class. Each row should sum to 1.0.
        labels: Array of shape (n_samples,) containing true class labels as integers.
        n_bins: Number of bins to partition confidence scores into. Default is 10.

    Returns:
        The Maximum Calibration Error as a float in the range [0, 1].
        Lower values indicate better calibration.

    Example:
        >>> probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
        >>> labels = np.array([0, 0, 1])
        >>> mce_score = mce(probs, labels, n_bins=10)
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = predictions == labels

    bin_bounds = np.linspace(0, 1, n_bins + 1)
    max_calibration_error = 0.0

    for i in range(n_bins):
        bin_lower = bin_bounds[i]
        bin_upper = bin_bounds[i + 1]

        if i == 0:
            in_bin_mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin_mask = (confidences > bin_lower) & (confidences <= bin_upper)

        proportion_in_bin = np.mean(in_bin_mask)

        if proportion_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin_mask])
            avg_confidence_in_bin = np.mean(confidences[in_bin_mask])
            calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_calibration_error = max(max_calibration_error, calibration_error)

    return max_calibration_error


def ace(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate the Adaptive Calibration Error (ACE) for probabilistic predictions.

    ACE addresses the bias introduced by fixed-bin ECE by using an adaptive binning
    strategy. Instead of fixed confidence intervals, bins are spaced to ensure each
    bin contains an equal number of samples. This is achieved by sorting predictions
    by confidence and dividing them into equal-sized groups.

    Args:
        probabilities: Array of shape (n_samples, n_classes) containing predicted probabilities
                      for each class. Each row should sum to 1.0.
        labels: Array of shape (n_samples,) containing true class labels as integers.
        n_bins: Number of bins to partition samples into. Default is 10.

    Returns:
        The Adaptive Calibration Error as a float in the range [0, 1].
        Lower values indicate better calibration.

    Example:
        >>> probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
        >>> labels = np.array([0, 0, 1])
        >>> ace_score = ace(probs, labels, n_bins=15)
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = predictions == labels

    n_samples = len(confidences)

    sorted_indices = np.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]

    samples_per_bin = n_samples // n_bins
    ace_value = 0.0

    for i in range(n_bins):
        start_idx = i * samples_per_bin
        if i == n_bins - 1:
            end_idx = n_samples
        else:
            end_idx = (i + 1) * samples_per_bin

        bin_confidences = sorted_confidences[start_idx:end_idx]
        bin_accuracies = sorted_accuracies[start_idx:end_idx]

        if len(bin_confidences) > 0:
            avg_confidence = np.mean(bin_confidences)
            avg_accuracy = np.mean(bin_accuracies)
            proportion_in_bin = len(bin_confidences) / n_samples
            ace_value += np.abs(avg_confidence - avg_accuracy) * proportion_in_bin

    return ace_value


def thresholded_ace(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.001,
    n_bins: int = 10,
) -> float:
    """
    Calculate the Thresholded Adaptive Calibration Error (Thresholded ACE).

    This variant of ACE only evaluates probabilities that are above a specified threshold,
    using adaptive binning to ensure equal samples per bin. This filters out infinitesimal
    softmax predictions that can wash out the calibration score, while maintaining the
    benefits of adaptive binning over fixed binning.

    Args:
        probabilities: Array of shape (n_samples, n_classes) containing predicted probabilities
                      for each class. Each row should sum to 1.0.
        labels: Array of shape (n_samples,) containing true class labels as integers.
        threshold: Minimum probability value to consider. Probabilities below this are excluded.
                  Default is 0.001.
        n_bins: Number of bins to partition samples into. Default is 10.

    Returns:
        The Thresholded Adaptive Calibration Error as a float in the range [0, 1].
        Lower values indicate better calibration.

    Example:
        >>> probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
        >>> labels = np.array([0, 0, 1])
        >>> tace_score = thresholded_ace(probs, labels, threshold=0.01, n_bins=15)
    """

    n_samples, n_classes = probabilities.shape
    all_probs = probabilities.flatten()
    all_labels = np.repeat(np.arange(n_classes), n_samples)
    true_labels = np.tile(labels, n_classes)

    above_threshold_mask = all_probs >= threshold
    filtered_probs = all_probs[above_threshold_mask]
    filtered_labels = all_labels[above_threshold_mask]
    filtered_true_labels = true_labels[above_threshold_mask]

    accuracies = filtered_labels == filtered_true_labels

    if len(filtered_probs) == 0:
        return 0.0

    n_filtered = len(filtered_probs)
    sorted_indices = np.argsort(filtered_probs)
    sorted_confidences = filtered_probs[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]

    samples_per_bin = n_filtered // n_bins
    ace_value = 0.0

    for i in range(n_bins):
        start_idx = i * samples_per_bin
        if i == n_bins - 1:
            end_idx = n_filtered
        else:
            end_idx = (i + 1) * samples_per_bin

        bin_confidences = sorted_confidences[start_idx:end_idx]
        bin_accuracies = sorted_accuracies[start_idx:end_idx]

        if len(bin_confidences) > 0:
            avg_confidence = np.mean(bin_confidences)
            avg_accuracy = np.mean(bin_accuracies)
            proportion_in_bin = len(bin_confidences) / n_filtered
            ace_value += np.abs(avg_confidence - avg_accuracy) * proportion_in_bin

    return ace_value


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the Brier Score for probabilistic predictions.

    The Brier Score is a proper scoring rule that measures both discrimination and
    calibration of probabilistic predictions. It computes the mean squared error between
    predicted probabilities and the true one-hot encoded labels.

    For multi-class problems, the Brier Score is computed as the sum of squared differences
    between predicted probabilities and binary indicators across all classes and samples.

    Args:
        probabilities: Array of shape (n_samples, n_classes) containing predicted probabilities
                      for each class. Each row should sum to 1.0.
        labels: Array of shape (n_samples,) containing true class labels as integers.

    Returns:
        The Brier Score as a float in the range [0, 1].
        Lower values indicate better calibration and discrimination, with 0 being perfect.

    Example:
        >>> probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
        >>> labels = np.array([0, 0, 1])
        >>> bs = brier_score(probs, labels)
    """
    n_samples, n_classes = probabilities.shape

    one_hot_labels = np.zeros((n_samples, n_classes))
    one_hot_labels[np.arange(n_samples), labels] = 1

    brier = np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=1))

    return brier
