from itertools import product
from fractions import Fraction
"""
Expression codes:
Addition: 1
Subtraction: -1
Multiplication: 2
Division: -2

Lower bound (proved with simple logic):
A000041

Upper bound (doesnt need proving):
4^(n-1)
"""
def calc_expres(val: int, operations: list[int]) -> float:
    total = Fraction(0)
    current = Fraction(val)

    for op in operations:
        if op == 2:  # Multiplication
            current *= val
        elif op == -2:  # Division
            current /= val  # Correct division
        elif op == 1 or op == -1:  # Addition or subtraction
            total += current
            current = Fraction(val if op == 1 else -val)

    total += current
    return total

for n in range(1,100):
  expressionsArray = [1, 2, -1, -2]

  all_operations = product(expressionsArray, repeat=n-1)

  results = set()
  for ops in all_operations:
      result = calc_expres(n, ops)
      results.add(result)

  print(f"n={n}: {len(results)}")

""""
Calculation for any value past n = 10 may be necessary. I'd love to see someone else try computing these or someone might make it run in the background
"""
