# Preference Elicitation for Multi-objective Combinatorial Optimization with Active Learning and Maximum Likelihood Estimation
Constructive Preference Elicitation on PC configuration and Prize-Collecting TSP.
For a given number of steps, it generates two random queries, evaluates the preferred one using a Bradley-Terry model of a simulated user, updates the weights and computes metrics. 

## Required libraries
- cpmpy
- gurobipy (requires licence)
- sklearn
- numpy
- matplotlib

## Data
- All data are in file data/
- Data for PC configuration are taken from https://github.com/stefanoteso/setmargin
- Data for PC-TSP are taken from https://github.com/BastianVT/EfficientSP

## Files content
- Main.py: defines parameters and train. Produces a figure with plots of metrics.
- train.py: implement the active learning loop
- user.py: class to simulate a user
- CPproblem.py: defines the class of CP problems, inherited by the class PCConfig (implemented with cpmpy)
- TSPproblem.py: adapted from https://github.com/BastianVT/EfficientSP (implemented with gurobipy)
- pref_utils.py: auxiliary functions (pool generation, metrics and plots)
- hptune.py: auxiliary to tune hyper-parameters

## Running
```
python Main.py --pb "PC" --relax True --lr 2 --update 'MLE' --qs 'ensemble' 
```
Parameters:
- pb (str): 'PC' or 'TSP'
- relax (bool): True to train on a pool of relaxed solutions
- lr (float): learning rate
- update (str): 'SP' or 'NCE' or 'MLE' or 'MLE\_online'
- qs (str): type of query selection ('ensemble' for UCB without cluster or 'cluster' for UCB with cluster, 'baseline' for standard Choice Perceptron, 'baseline-relax' for Choice Perceptron on a pool of relaxed solutions)
- alpha: UCB trade-off. If alpha in \[0, 1\], then gamma = alpha. If alpha = 4, gamma = 1/t. 
