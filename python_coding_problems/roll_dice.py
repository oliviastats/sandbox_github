import numpy as np
def roll_dice(n: int):
    results = []
    for i in range(n):
        dice = np.random.randint(1,7)
        results.append(dice)
    dict_results = dict()
    for el in list(set(results)):
        dict_results[el] = results.count(el)
    return dict_results