import sys
sys.setrecursionlimit(2147483647)

def append_co_prime(n,arr):
  ret=0
  for i in arr:
    if n%i==0:
      ret=1
      break
  if ret==0:
    arr.append(n)
    return arr
  else:
    return arr

def is_co_prime(n,arr):
  ret=0
  for i in arr:
    if n%i==0:
      ret=1
      break
  if ret==0:
    return True
  else:
    return False

def primes_to(n):
  if n<2:
    return []
  if n==2:
    return [2]
  return append_co_prime(n,primes_to(n-1))

def is_prime(n):
  for i in primes_to(int(n**(1/2))):
    if n%i==0:
      return False
  return True
