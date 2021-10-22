def find_median(arr: list):
    ordered_arr = []
    while len(arr)>0:
        min_curr = min(arr)
        index = arr.index(min_curr)
        arr = arr[:index] + arr[index+1:]
        ordered_arr.append(min_curr)
    if len(arr)%2 == 0:
        middle_num = int(len(arr)/2)
        median = (ordered_arr[middle_num+1] + ordered_arr[middle_num+2])/2
    else:
        median_index = int((len(arr)-1)/2)
        median = ordered_arr[median_index+1]
    return median

def find_median2(arr: list):
    ordered_arr = sorted(arr)
    if len(ordered_arr)%2 == 0:
        median_index = int(len(ordered_arr)-1)
        median = (ordered_arr[median_index]+ordered_arr[median_index+1])/2
    else:
        median_index = int((len(ordered_arr)-1)/2)
        median = ordered_arr[median_index]
    return median 