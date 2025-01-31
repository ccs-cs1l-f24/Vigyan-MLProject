import random
import numpy

class RandomPlayer():
    def __init__(self):
        pass
    def action(self, valid_moves):
        choices = []
        for i in range(len(valid_moves)):
            if valid_moves[i]==1:
                choices.append(i)
        return choices[random.randint(0,len(choices)-1)]