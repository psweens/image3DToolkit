import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculate the Dice Coefficient between two 3D binary arrays.

    Args:
        y_true (np.ndarray): Ground truth binary 3D array.
        y_pred (np.ndarray): Predicted binary 3D array.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient score.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    intersection = np.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + smooth)

def jaccard_index(y_true, y_pred):
    """
    Calculate the Jaccard Index (Intersection over Union) for two 3D binary arrays.

    Args:
        y_true (np.ndarray): Ground truth binary 3D array.
        y_pred (np.ndarray): Predicted binary 3D array.

    Returns:
        float: Jaccard Index score.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
    return intersection / union

def precision(y_true, y_pred):
    """
    Calculate the precision score between two 3D binary arrays.

    Args:
        y_true (np.ndarray): Ground truth binary 3D array.
        y_pred (np.ndarray): Predicted binary 3D array.

    Returns:
        float: Precision score.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    true_positives = np.sum(y_true_flat * y_pred_flat)
    predicted_positives = np.sum(y_pred_flat)
    return true_positives / (predicted_positives + 1e-6)

def recall(y_true, y_pred):
    """
    Calculate the recall score between two 3D binary arrays.

    Args:
        y_true (np.ndarray): Ground truth binary 3D array.
        y_pred (np.ndarray): Predicted binary 3D array.

    Returns:
        float: Recall score.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    true_positives = np.sum(y_true_flat * y_pred_flat)
    possible_positives = np.sum(y_true_flat)
    return true_positives / (possible_positives + 1e-6)

def f1_score(y_true, y_pred):
    """
    Calculate the F1 score between two 3D binary arrays.

    Args:
        y_true (np.ndarray): Ground truth binary 3D array.
        y_pred (np.ndarray): Predicted binary 3D array.

    Returns:
        float: F1 score.
    """
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    return 2 * (precision_score * recall_score) / (precision_score + recall_score + 1e-6)

def sensitivity(y_true, y_pred):
    """
    Calculate the accuracy score between two 3D binary arrays.

    Args:
        y_true (np.ndarray): Ground truth binary 3D array.
        y_pred (np.ndarray): Predicted binary 3D array.

    Returns:
        float: Accuracy score.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    correct_predictions = np.sum(y_true_flat == y_pred_flat)
    total_elements = y_true_flat.size
    return correct_predictions / total_elements

def specificity(y_true, y_pred):
    """
    Calculate the specificity score between two 3D binary arrays.

    Args:
        y_true (np.ndarray): Ground truth binary 3D array.
        y_pred (np.ndarray): Predicted binary 3D array.

    Returns:
        float: Specificity score.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    true_negatives = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    possible_negatives = np.sum(y_true_flat == 0)
    return true_negatives / (possible_negatives + 1e-6)

def print_metrics(filename, y_true, y_pred, operations=None):
    if operations is None:
        operations = [f1_score, dice_coefficient, jaccard_index, specificity, sensitivity]
    print('%s ' %filename)
    for op in operations:
        print('%f ' %op(y_true, y_pred))