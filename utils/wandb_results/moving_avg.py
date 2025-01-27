import numpy as np


def moving_average(input_array, M):
    """
    Calculate the moving average of M consecutive entries in the 2D input array.

    Parameters:
    - input_array: NumPy array of shape (B, N)
    - M: Number of consecutive entries for calculating the moving average

    Returns:
    - result_array: NumPy array containing the moving averages for each row
    """
    B, N = input_array.shape

    # Check if M is valid
    if M <= 0 or M > N:
        raise ValueError("Invalid value of M")

    # Use convolution to calculate the moving average along the last axis (axis=1)
    kernel = np.ones(M) / M
    result_array = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), axis=1, arr=input_array
    )

    # Concatenate the first M - 1 Dimensions
    prefix = np.zeros([input_array.shape[0], M - 1])
    for i in range(1, M):
        prefix[:, i - 1] = input_array[:, :i].mean(1)
    result_array = np.concatenate([prefix, result_array], 1)
    return result_array
