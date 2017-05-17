from functools import reduce
from math import factorial, pow
from operator import mul


def choose(n, i):
    return factorial(n)/(factorial(i) * factorial(n-i))


def num_policies(num_conditions):
    return reduce(mul,
                  (pow(num_conditions - i, choose(num_conditions, i))
                   for i in range(num_conditions))
                  )

if __name__ == '__main__':
    for i in range(2, 10):
        print(i, num_policies(i))