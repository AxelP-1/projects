"""Expressions:
Addition: 1;
Substraction: -1;
Multiplication: 2
Division: -2;
"""
def calc_expres(val, operations):
    tot = 0
    curr = val
    i = 0

    while i < len(operations):
        op = operations[i]

        if op == 2:
            curr *= val
        elif op == -2:
            curr /= val
        elif op == 1 or op == -1:
            tot += curr
            curr = val if op == 1 else -val
        i += 1

    tot += curr
    return tot

n=3

expressionsArray=[1,2,-1,-2]
solutions=[]
for i in range(4**(n-1)):
  currOperations=[]
  for j in range(n-1):
    currOperations.append(expressionsArray[(i // (4 ** j)) % 4])
  solutions.append(calc_expres(n,currOperations))

seen = {}

for i in solutions:
  if seen.get(i,0)==0:
    seen[i]=1

print(len(seen))
