def merge_two_lists(l1: list, l2: list):
    output = []
    if len(l1) == len(l2):
        for i in range(len(l1)):
            if l1[i] <= l2[i]:
                output.append(l1[i])
                output.append(l2[i])
            else:
                output.append(l2[i])
                output.append(l1[i])
    elif len(l1) > len(l2):
        output = l1
    else:
        output = l2
    return output