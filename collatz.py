import matplotlib.pyplot as plt


col = {1: 0}

def calc(x):
    if x in col:
        return col[x]
    if x % 2 == 0:
        next_x = x // 2
    else:
        next_x = 3 * x + 1
    col[x] = calc(next_x) + 1
    return col[x]

mx = -1
b = 0
rcrdbrk=[]
while True:
    b += 1
    steps = calc(b)
    if steps > mx:
        mx = steps
        rcrdbrk.append(b)
        print(f"n={b}: {steps}")
        """
        plt.plot(rcrdbrk)
        plt.show()
        """
