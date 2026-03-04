# Markov-Decision-Processes (Dynamic Programming)

This repository contains the solution for the Dynamic Programming (Markov Decision Processes) exercise from the **Planning and Decision Making for Autonomous Robots (PDM4AR)** master's course at **ETH Zurich** (Fall 2025). 

## Overview
This project tackles the challenge of calculating an optimal policy and value function for an autonomous survey robot operating on a distant planet. The environment is modeled as a stationary Markov Decision Process (MDP) over a discrete $M \times N$ grid. The ultimate goal is to maximize the expected profit while navigating various terrain types and dealing with stochastic transitions (e.g., slipping in swamps or unpredictable teleports).

## The Problem
* **Objective:** Compute an optimal policy for the robot to reach a `GOAL` location from a `START` base while maximizing profit.
* **Challenges:** Modeling the transition probabilities properly for different terrain types and handling stochasticity in movement (success probabilities vs splits vs breakdown events) and applying both Value and Policy Iteration correctly.

### The World (Grid)
The grid consists of various cell types that affect the robot's movement and rewards:
*   **`GRASS`**: Standard terrain, taking 1 hour to cross. Success probability is 0.75, with a 0.25 chance of slipping into other adjacent directions.
*   **`SWAMP`**: Difficult terrain, taking 2 hours to cross. Success probability is 0.5, with 0.25 split for other directions, a 0.2 chance of staying in place, and a 0.05 chance of breaking down.
*   **`WONDERLAND`**: Acts like grass but teleports the robot instantly to a random adjacent cell. Provides a massive **+3k USD** reward (as long as it doesn't drop the robot onto a cliff).
*   **`CLIFF`**: Untraversable. Moving into it causes an immediate breakdown.
*   **`GOAL`**: The destination. Reaching and staying here provides a **+50k USD** reward.
*   **`START`**: The base where new robots are deployed.

### Rewards and Costs
The MDP is built on maximizing monetary profit. Time is money, and deploying robots isn't cheap:
*   **Goal Bonus**: +50k USD for reaching and choosing the `STAY` action at the goal.
*   **Time Compensation**: -1k USD for every hour spent traversing the grid.
*   **Deployment Cost**: -10k USD for every replacement robot deployed at `START` after a breakdown (the first one is free).
*   **Wonderland Reward**: +3k USD for successfully traversing a wonderland cell.

## Performance Results

Our algorithms were rigorously tested against **21 automated test cases** and achieved a **100% completion rate** with zero exceptions. Furthermore, our highly optimized dynamic programming solvers dramatically outperformed the course's reference benchmarks:

| Metric | Our Implementation | Course Reference |
| :--- | :---: | :---: |
| **Transition Probability Accuracy** | 100% (1.0) | 1.0 |
| **Value Iteration Accuracy ($R^2$)** | 100% (1.0) | 1.0 |
| **Value Iteration Solve Time** | **1.432s** | 8.714s |
| **Policy Iteration Accuracy** | 100% (1.0) | 1.0 |
| **Policy Iteration Solve Time** | **1.295s** | 4.263s |

*Note: Our optimized Policy Iteration algorithm runs roughly **3.3x faster** than the course reference, and our Value Iteration algorithm runs over **6x faster**.*

## Our Approach

We solved the Markov Decision Process using two distinct Dynamic Programming algorithms, both of which calculate the mathematically optimal action from any given state in the grid based on the transition probabilities and a discount factor $\gamma = 0.9$.

### 1. MDP Definition (`mdp.py`)
Before solving, the exact Transition Model $P(s' | s, a)$ and Expected Reward Model $R(s, a, s')$ had to be mathematically defined. This required meticulously calculating the edge cases for grid boundaries, the cascading probabilities of stochastic slippage into a cliff or a wonderland cell, and the exact step-costs for each action transition.

### 2. Value Iteration (`value_iteration.py`)
We implemented the standard Value Iteration algorithm:
* The algorithm iteratively computes the exact Value Function $V(s)$ for every state by performing the Bellman Update: 
  $$V_{k+1}(s) = \max_a \sum_{s'} P(s' | s, a) [R(s, a, s') + \gamma V_k(s')]$$
* It terminates once the maximum change in value across all states drops below a defined threshold $\epsilon$. 
* Finally, it extracts the deterministic optimal Policy $\pi(s)$ by acting greedily with respect to the converged Value Function.

### 3. Policy Iteration (`policy_iteration.py`)
We implemented the full Policy Iteration algorithm to directly compute the optimal policy without waiting for the Value Function to iteratively converge completely:
* **Policy Evaluation:** Given a current policy $\pi$, we compute its exact expected returns $V^\pi(s)$ by solving the linear system of equations defined by the Bellman Expectation Equation.
* **Policy Improvement:** We then act greedily with respect to the newly evaluated $V^\pi(s)$ to generate a new policy $\pi'$. 
* This cycle repeats until the policy stops changing. Due to the exact nature of the evaluation step, Policy Iteration often converges in fewer steps than Value Iteration.

## Setup and Execution
To guarantee execution regardless of your host operating system or local Python dependencies, this project is configured to run entirely inside a **VS Code Devcontainer**.

### Recommended Installation (Docker + VS Code)
1. Ensure you have **Docker Desktop** running and **Visual Studio Code** installed with the "Dev Containers" extension.
2. Clone this repository and open it in VS Code.
3. VS Code will detect the `.devcontainer` folder and prompt you to **"Reopen in Container"**. Click it. (If it doesn't prompt, press `Ctrl+Shift+P` and type `Dev Containers: Reopen in Container`).
4. Docker will automatically build the environment and install all dependencies via Poetry.

### Running the Code
Once inside the Devcontainer, you have two simple ways to execute the simulation:

**Option 1: Using the VS Code Debugger (Recommended)**
* Open the "Run and Debug" panel in VS Code (`Ctrl+Shift+D`).
* Select the pre-configured **"Exercise04 - Run"** or **"Exercise04 - Debug"** profile and hit play.

**Option 2: Using the Terminal**
Run the built-in PDM4AR course CLI script directly:
```bash
poetry run python src/pdm4ar/main.py -e 04
```

## Acknowledgements
This codebase was extracted and adapted from the final project of the **Planning and Decision Making for Autonomous Robots (PDM4AR)** master's course at **ETH Zurich** (Fall 2025). The underlying exercise framework, simulation grid, and evaluation metrics were provided by the [IDSC Frazzoli Lab](https://github.com/PDM4AR). You can find the original exercise description and constraints [here](https://pdm4ar.github.io/exercises/04-dynamicprogramming.html).
