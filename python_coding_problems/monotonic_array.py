import numpy as np

def check_if_monotnic_array(array_input: np.array):
    results = np.array([])
    for i in range(len(array_input)-1):
        results = np.append(results, array_input[i+1] - array_input[i])

    if ((len(results[results >= 0]) == len(results)) | (len(results[results <= 0]) == len(results))):
        return True
    else: 
        return False
