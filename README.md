# This is my ML project for CCS CS 1L Fall 2024
The goal of this project is to recreate the AlphaZero neural network implementation to run on different games with the end goal being to analyze the Game of Cycles to better understand how to play the game for potential future research on Cycles.

# Descripition

## Game Files

### Tic Tac Toe
- How to play : [Tic Tac Toe](https://en.wikipedia.org/wiki/Tic-tac-toe)
- Just read the code I don't got da time for dis

### Connect Four
- How to play : [Connect Four](https://en.wikipedia.org/wiki/Connect_Four)

- If I didn't do the Tic Tac Toe how da heck am i gonna do this

### Cycles
- How to play : [Cycles](https://arxiv.org/abs/2004.00776)

- Ok I might actually do this one later who knows tho

## AlphaZero & Supporting Files
I haven't done these yet, but I'll hopefully get to them
### MCTS

### MCTSParallel

### ResNet

### AlphaZero

### AlphaZero Parallel

## Testfiles

### testTicTacToe

### testConnect4

### testCycles

### AlphaZeroTest

# Getting Started with the Code

## Dependencies

- Python version 3.12.7, whatever you have must be compatible with PyTorch
    - I got it from their [website](https://www.python.org/downloads/), but I'm sure you can use pip or something
- NumPy version 2.1.2
  ```
  pip install numpy
  ```
    - I used pip for this one, I doubt any different version would break the code

- PyTorch version 2.5.0
    - I got this from their [website](https://pytorch.org/get-started/locally/)
- tqdm version 4.66.5 (Kinda optional tho, but has sick bars)
    ```
    pip install tdqm
    ```

## Installing
To get the code clone the repository:
```
git clone https://github.com/ccs-cs1l-f24/Vigyan-MLProject.git
```

If you are on a not mac, you might need to change the GPU that PyTorch searches for, this command is as follows and appears throughout the code, I aint finding them that up to you gl lmao:
```
device = torch.device("mps" if torch.backends.mps.is_available() else "CPU")
```

Change "mps" to "cuda" if you have an NVIDIA GPU. If you have some other thing use Google. If you are lazy and don't care, it shouldn't matter as it will use the CPU if it doesn't support your GPU, the only downside is missing out on a ~300% speed increase at least on my Mac for training.
## How to run specific games

### Train the model
Navigate to the AlphaZeroTest.py, and change the following line of code to the game you want:
```
game = [insert game here]
```
```
EX: game = Cycles.Cycles(adj_matrix=adj_matrix, valid_cycles=valid_cycles)
EX: game = TicTacToe.TicTacToe()
```

Depending on the game you might have to create input for it, like, for the example of Cycles, it needs to know the adjacency matrix and the list of valid cycles on the map. But, a game like Tic Tac Toe doesn't require any input.
You also need to change the 'directory' argument in args to be where you want the model to be stored.
```
'directory' = "[file path here]"
```

Run the file with the following command:
```
python3.12 AlphaZeroTest.py
```

Once it finishes running, which might take a bit, 
### Play the game
Navigate to the game's test.py file. In the args ensure that the args are the same as in the AlphaZeroTest.py file except for the 'dirichlet_epsilon' which should be set to 0. The parameter 'trained_model' should be set to the path to the model that you trained.
```
'dirichlet_epsilon' = 0,
'trained_model' = [path to model location]
```
Once these are set you can run the file and have fun playing against the AI!
```
python3.12 [filename]
```

You can change who goes first by changing the following command where 1: Player goes first, -1: AI goes first:
```
while True:
    print(state)
    if player==[Value = 1 or -1]:
```
