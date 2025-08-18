from random import choice, random
import matplotlib.pyplot as plt

BOARD = [2, 5, 3, 30, 2, 3, 30, 5, 30]
PLAYERS = 3

def tell_me_what_you_see(hide, creatures):
    ret = [[], [], [], [], [], [], [], [], []]
    for i in range(9):
        for j in range(len(creatures[i])):
            if hide[i] % creatures[i][j] == 0:
                ret[i].append(creatures[i][j])
            else:
                ret[8].append(creatures[i][j])
    return ret

def roll_over_beetoven():
    diceFaces = [1, 1, 2, 2, "loup", 3]
    return diceFaces[int(random() * 6)]

def the_long_and_winding_road(strategy, board, players):
    nbTurn = 0
    end = 3
    creatures = [[], [], [], [], [], [], [], [], [2, 3, 5]]
    while end > 0:
        rol = roll_over_beetoven()
        if rol != "loup":
            creatures = strategy(rol, creatures, board)
        else:
            creatures = tell_me_what_you_see(board, creatures)

        for i in range(players - 1):
            rol = roll_over_beetoven()
            if rol == "loup":
                creatures = tell_me_what_you_see(board, creatures)
                break

        end = 0
        for i in range(len(creatures)):
            end += len(creatures[i])
        nbTurn += 1

    return nbTurn

def dizzy_miss_lizzy(roll, creatures, board):
    all_creatures = []
    for i in range(len(creatures)):
        for j in range(len(creatures[i])):
            all_creatures.append((i, creatures[i][j]))

    if len(all_creatures) == 0:
        return creatures

    chosen = choice(all_creatures)
    old_pos = chosen[0]
    creature = chosen[1]

    new_creatures = []
    for group in creatures:
        new_creatures.append(group[:])

    new_creatures[old_pos].remove(creature)
    new_pos = old_pos - roll
    if new_pos >= 0:
        new_creatures[new_pos].append(creature)
    return new_creatures

def hello_goodbye(roll, creatures, board):
    new_creatures = []
    for group in creatures:
        new_creatures.append(group[:])

    for pos in range(len(creatures)):
        if len(creatures[pos]) > 0:
            creature = creatures[pos][0]
            new_creatures[pos].remove(creature)
            new_pos = pos - roll
            if new_pos >= 0:
                new_creatures[new_pos].append(creature)
            break
    return new_creatures

def come_together(roll, creatures, board):
    new_creatures = []
    for group in creatures:
        new_creatures.append(group[:])

    for pos in range(len(creatures)-1, -1, -1):
        if len(creatures[pos]) > 0:
            creature = creatures[pos][0]
            new_creatures[pos].remove(creature)
            new_pos = pos - roll
            if new_pos >= 0:
                new_creatures[new_pos].append(creature)
            break
    return new_creatures

def revolution(roll, creatures, board):
    new_creatures = []
    for group in creatures:
        new_creatures.append(group[:])

    for pos in range(len(creatures)-1, -1, -1):
        if len(creatures[pos]) > 0:
           for i in range(len(creatures[pos])):
              if board[pos]%creatures[pos][i]!=0:
                creature = creatures[pos][i]
                new_creatures[pos].remove(creature)
                new_pos = pos - roll
                if new_pos >= 0:
                    new_creatures[new_pos].append(creature)
                return new_creatures

    for pos in range(len(creatures)-1, -1, -1):
        if len(creatures[pos]) > 0:
           for i in range(len(creatures[pos])):
              creature = creatures[pos][i]
              if board[pos-roll]%creature==0:
                new_creatures[pos].remove(creature)
                new_pos = pos - roll
                if new_pos >= 0:
                    new_creatures[new_pos].append(creature)
                return new_creatures
    

    return come_together(roll,creatures,board)

def revolution9(roll, creatures, board):
    new_creatures = []
    for group in creatures:
        new_creatures.append(group[:])

    for pos in range(len(creatures)):
        if len(creatures[pos]) > 0:
           for i in range(len(creatures[pos])):
              if board[pos]%creatures[pos][i]!=0:
                creature = creatures[pos][i]
                new_creatures[pos].remove(creature)
                new_pos = pos - roll
                if new_pos >= 0:
                    new_creatures[new_pos].append(creature)
                return new_creatures

    for pos in range(len(creatures)):
        if len(creatures[pos]) > 0:
           for i in range(len(creatures[pos])):
              creature = creatures[pos][i]
              if board[pos-roll]%creature==0:
                new_creatures[pos].remove(creature)
                new_pos = pos - roll
                if new_pos >= 0:
                    new_creatures[new_pos].append(creature)
                return new_creatures
    

    return hello_goodbye(roll,creatures,board)

def obladi_oblada(roll, creatures, board):
    new_creatures = []
    for group in creatures:
        new_creatures.append(group[:])

    for pos in range(len(creatures)-1, -1, -1):
        if len(creatures[pos]) > 0:
           for i in range(len(creatures[pos])):
              if board[pos]%creatures[pos][i]!=0:
                creature = creatures[pos][i]
                new_creatures[pos].remove(creature)
                new_pos = pos - roll
                if new_pos >= 0:
                    new_creatures[new_pos].append(creature)
                return new_creatures

    for pos in range(len(creatures)-1, -1, -1):
        if len(creatures[pos]) > 0:
           for i in range(len(creatures[pos])):
              creature = creatures[pos][i]
              if board[pos-roll]%creature==0:
                new_creatures[pos].remove(creature)
                new_pos = pos - roll
                if new_pos >= 0:
                    new_creatures[new_pos].append(creature)
                return new_creatures
      
    return hello_goodbye(roll, creatures, board)

plt.figure(figsize=(12, 8))

strategies = [dizzy_miss_lizzy, #alleatoire
              come_together,    #bouger le dernier
              hello_goodbye,    #le premier
              revolution,       #prendre en compte les cachettes en priorisant les premiers
              revolution9,      #prendre en compte les cachettes en priorisant les premiers
              obladi_oblada    #same as prev 2
]
names = ['Dizzy Miss Lizzy', 'Come Together', 'Hello, Goodbye', 'Revolution', 'Revolution 9','Obladi Oblada']

colours = ["red","green","blue","black","orange","purple","yellow","magenta"]

colours.sort()

for i in range(len(strategies)):
    strategy = strategies[i]
    name = names[i]

    results = []
    for _ in range(10000):
        results.append(the_long_and_winding_road(strategy, BOARD, PLAYERS))

    total = 0
    for r in results:
        total += r
    mean = total / len(results)

    bins = [0] * 200
    for r in results:
        if r < 200:
            bins[r] += 1

    x = list(range(200))
    color = colours[i]
    plt.plot(x, bins, label=name, color=color)
    plt.axvline(mean, color=color, linestyle='--')

plt.xlabel('number of turns')
plt.ylabel('frequency')
plt.title('strategy performance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
thanks to the beatles for their memorable and diverse names
thanks to the people who think variables cannot be a single letter, now I name them by even less explanatory song names
"""
