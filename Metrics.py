# Import required dependencies
import numpy as np


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """Implementation of the Mean Absolute Error (MAE)"""
    ### TO BE COMPLETED BY THE STUDENTS ###

    return np.mean(np.abs(y_true - y_pred))


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """Implementation of the Mean Squared Error (MSE)"""
    ### TO BE COMPLETED BY THE STUDENTS ###

    return np.mean((y_true - y_pred) ** 2)


def R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """Implementation of the R2 metric"""
    ### TO BE COMPLETED BY THE STUDENTS ###

    y_mean = np.mean(y_true)

    # Numerator: Sum of squared errors between real and prediction values
    ss_real_predicted = np.sum((y_true - y_pred) ** 2)

    # Denominator: Sum of squared errors between real and mean values
    ss_real_mean = np.sum((y_true - y_mean) ** 2)

    return 1 - (ss_real_predicted / ss_real_mean)


def Corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """Implementation of the Pearson's Correlation Coefficient"""
    ### TO BE COMPLETED BY THE STUDENTS ###

    # corrcoef returns a 2x2 matrix
    # [[corr(y,y),    corr(y,ŷ)],
    # [corr(ŷ,y),   corr(ŷ,ŷ)]]

    return np.corrcoef(y_true, y_pred)[0, 1]
