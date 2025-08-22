from itertools import product
from fractions import Fraction
"""
Expression codes:
Addition: 1
Subtraction: -1
Multiplication: 2
Division: -2
"""
def calc_expres(val: int, operations: list[int]) -> float:
    total = Fraction(str(0))
    current = Fraction(str(val))

    for op in operations:
        if op == 2:
            current *= Fraction(str(val))
        elif op == -2:
            current = Fraction(current,val)
        elif op == 1 or op == -1:
            total += Fraction(str(current))
            current = Fraction(str(val)) if op == 1 else Fraction(str(-val))

    total += current
    return total
for n in range(15,100):
  expressionsArray = [1, 2, -1, -2]

  all_operations = product(expressionsArray, repeat=n-1)

  results = set()
  for ops in all_operations:
      result = calc_expres(n, ops)
      results.add(result)

  print(f"n={n}: {len(results)}")
