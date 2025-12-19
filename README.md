# This is my ML project for CCS CS 1L Fall 2024

The goal of this project is to recreate and study the **AlphaZero** reinforcement learning algorithm in a modular and extensible way. The implementation is designed to run on multiple perfect‑information, turn‑based games, with the long‑term research objective of analyzing the **Game of Cycles** to better understand optimal strategies and emergent structure.  

Unlike supervised learning approaches, this project focuses entirely on **self‑play reinforcement learning**, where agents learn purely from playing against themselves, without human demonstrations or handcrafted heuristics.

---

# Description

## Game Files

Each game is implemented as its own Python module following a shared interface. This allows the AlphaZero core to interact with any game in a generic way, as long as the game defines:
- A representation of the game state
- Legal actions from a state
- State transition logic
- Terminal state detection and reward assignment

### Tic Tac Toe

- How to play: https://en.wikipedia.org/wiki/Tic-tac-toe

This is the simplest environment in the project and serves as a **sanity‑check game** for validating the AlphaZero pipeline. Because the state space is small and the optimal strategy is known, Tic Tac Toe is useful for:
- Verifying MCTS correctness
- Testing neural network convergence
- Debugging training stability issues

The implementation encodes the board state numerically and supports move validation, terminal detection (win/loss/draw), and reward propagation.

---

### Connect Four

- How to play: https://en.wikipedia.org/wiki/Connect_Four

Connect Four significantly increases complexity compared to Tic Tac Toe:
- Larger board
- Deeper game tree
- More meaningful long‑term planning

This game is used to test whether the AlphaZero implementation scales beyond trivial environments. Training requires more MCTS simulations and longer self‑play sessions to reach competent play.

---

### Cycles

- How to play: https://arxiv.org/abs/2004.00776

The **Game of Cycles** is the primary research motivation for this project. Unlike classic grid‑based games, Cycles is played on a graph structure defined by:
- An adjacency matrix
- A predefined set of valid cycles

The game module accepts:
- An adjacency matrix defining the graph
- A list of valid cycles present in the graph

This makes Cycles an ideal testbed for studying AlphaZero on **non‑spatial, graph‑structured environments**, which are less explored in standard AlphaZero literature.

---

## AlphaZero & Supporting Files

These files implement the core AlphaZero algorithm and its supporting components.

### MCTS

The **Monte Carlo Tree Search (MCTS)** module performs guided tree search using the neural network’s policy and value predictions.

Key features:
- Upper Confidence Bound (PUCT) selection
- Expansion using neural network priors
- Backpropagation of value estimates
- Temperature‑controlled action selection during training

MCTS is responsible for balancing exploration and exploitation during self‑play and producing improved policy targets for training.

---

### MCTSParallel

This module extends the base MCTS to support **parallel simulations**, allowing multiple rollouts to be performed simultaneously. Parallelization significantly speeds up training, especially for larger games like Connect Four and Cycles.

---

### ResNet

The neural network is implemented as a **Residual Network (ResNet)**, following the architecture introduced in AlphaGo Zero.

Structure:
- Shared convolutional trunk
- Policy head (predicts move probabilities)
- Value head (predicts expected outcome)

Residual connections help stabilize training and allow deeper networks without vanishing gradients.

---

### AlphaZero

This file contains the main AlphaZero training loop, including:
- Self‑play game generation
- MCTS‑guided move selection
- Replay buffer management
- Neural network training on self‑play data

The model is trained to minimize:
- Policy loss (cross‑entropy with MCTS visit distribution)
- Value loss (mean squared error)
- Optional regularization terms

---

### AlphaZeroParallel

This is a parallelized version of the AlphaZero training loop, designed to:
- Generate self‑play games concurrently
- Improve GPU utilization
- Reduce overall training time

---

## Bayesian Optimization

Training AlphaZero involves many hyperparameters that strongly affect performance and stability. Manually tuning these values is inefficient and unreliable, so this project incorporates **Bayesian Optimization** to automate hyperparameter search.

### What Is Being Optimized

Typical parameters optimized include:
- Learning rate
- Number of MCTS simulations per move
- Exploration constant (PUCT)
- Dirichlet noise parameters
- Training batch size

### Why Bayesian Optimization

Bayesian Optimization is well‑suited for AlphaZero because:
- Each training run is expensive
- The objective function is noisy
- The relationship between parameters and performance is non‑linear

A probabilistic surrogate model (e.g., Gaussian Process) is used to model performance and intelligently select new hyperparameter configurations that balance exploration and exploitation.

### Integration Into This Project

The optimization loop:
1. Selects a set of hyperparameters
2. Trains AlphaZero for a fixed budget
3. Evaluates performance (e.g., win rate or value loss)
4. Updates the surrogate model
5. Proposes improved parameters

This allows the system to converge on strong configurations faster than grid or random search. We use multithreading to evaluate millions of potential hyperparameters in the surrogate model.

---

## Test Files

Test files are provided to validate correctness and functionality.

### testTicTacToe
Verifies game logic, terminal conditions, and valid move generation.

### testConnect4
Tests board updates, win detection, and action legality.

### testCycles
Ensures correct graph traversal, cycle validation, and reward assignment.

### AlphaZeroTest
Primary entry point for training and evaluation.

---

# Getting Started with the Code

## Dependencies

- Python 3.12.7 (must be compatible with PyTorch)
- NumPy 2.1.2
```bash
pip install numpy
```
- PyTorch 2.5.0 (install from https://pytorch.org)
- tqdm 4.66.5 (optional, progress bars)
```bash
pip install tqdm
```

---

## Installing

Clone the repository:
```bash
git clone https://github.com/ccs-cs1l-f24/Vigyan-MLProject.git
```

### Device Configuration

The code defaults to Apple Silicon GPU acceleration:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

For NVIDIA GPUs, change `"mps"` to `"cuda"`.  
If unsupported, the code will fall back to CPU execution.

---

## How to Run Specific Games

### Train the Model

Open `AlphaZeroTest.py` and select a game:
```python
game = TicTacToe.TicTacToe()
# or
game = Cycles.Cycles(adj_matrix=adj_matrix, valid_cycles=valid_cycles)
```

Set the output directory for the model weights:
```python
'directory' = "[file path here]"
```

Run training:
```bash
python3.12 AlphaZeroTest.py
```

---

### Play the Game

Navigate to the game’s test file and update:
```python
'dirichlet_epsilon' = 0
'trained_model' = "[path to trained model]"
```

Run:
```bash
python3.12 [filename]
```

Change who moves first:
```python
if player == 1:
    # Player first
else:
    # AI first
```

---

## Acknowledgments

Use was made of the computational facilities administered by the Center for Scientific Computing at the CNSI and MRL (an NSF MRSEC; DMR‑2308708) and purchased through NSF CNS‑1725797.
