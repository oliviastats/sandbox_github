def fibonacci_number(n: int):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_number(n-2)+fibonacci_number(n-1)

def fibonacci_seq(n: int):
    fibonacci = []
    for i in range(n):
        fibonacci.append(fibonacci_number(i))
    return fibonacci
