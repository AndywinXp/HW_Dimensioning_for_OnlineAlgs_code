# Constrained Hardware Dimensioning for Online Algorithms (code)

This repository contains the code used for the [experimental evaluation](https://github.com/AndywinXp/HW_Dimensioning_for_OnlineAlgs).

We propose an approach for automatic hardware dimensioning and algorithm configuration of two state-of-the-art online algorithms under an heterogeneous set of constraints.
We rely on the integration of Machine Learning models within an optimization problem, following the Empirical Model Learning paradigm by Lombardi et al, "Empirical decision model learning", 2017. Machine Learning is used for benchmarking the target algorithm on different hardware configurations and optimization is used to find the optimal matching of computing resources and algorithm configuration, while respecting user-defined constraints (e.g., cost, time, solution quality). The empirical evaluation shows that our method can find optimal combinations, in terms of algorithm configuration and hardware dimensioning, for new, unseen data instances, while its flexibility allows the adoption of different constraints and/or objectives, as required by users.

## Approach Detail

We propose an optimization method composed of an optimization model and two sets (one for each of the algorithms) of three ML regression models, each one devoted to predicting a specific target, given a certain input instance and a certain value for the decisional variables of the algorithms; the three regression targets are the 1) time required by the online heuristic to find a solution, 2) the amount of memory, and 3) the solution quality. These ML models are then embedded in the optimization problem, which has the purpose of providing the optimal combination in terms of algorithm configuration and hardware dimensioning, given a set of user-specified constraints (e.g., bounding the run-time of the algorithm of obtaining solutions with quality higher than a threshold).

## Repository Content
Each directory in the repository corresponds to a section of the paper; each of them contains the script to build the EML model, the script to execute the experiments and the scripts used to generate the graphs, where applicable. The only exception to this is the folder `4.1 - ML Models Exploration`, which contains the scripts used to generate and validate the ML models used throughout the other experimentations phases.
