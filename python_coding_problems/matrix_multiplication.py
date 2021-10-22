def matrix_multiplication(A: list, B: list):
    result = [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result