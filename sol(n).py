from itertools import product
"""
Expression codes:
Addition: 1
Subtraction: -1
Multiplication: 2
Division: -2
"""
def calc_expres(val: int, operations: list[int]) -> float:
    total = 0
    current = val
    
    for op in operations:
        if op == 2:
            current *= val
        elif op == -2:
            current /= val
        elif op == 1 or op == -1: 
            total += current
            current = val if op == 1 else -val
    
    total += current
    return total

n = 12
expressionsArray = [1, 2, -1, -2]

all_operations = product(expressionsArray, repeat=n-1)

results = set()
for ops in all_operations:
    result = calc_expres(n, ops)
    results.add(result)

print(len(results))
