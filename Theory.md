# Conventional Vs Enhanced Random Forests for Handling Imbalanced Data

Conventional Random Forest implementations do not typically include the specific techniques for handling imbalanced data internally. Here's a breakdown of how to build enhanced random forest for handling imbalanced data:

## 1. Class Weights

**Conventional Random Forest:**
Does not typically use class weights unless explicitly specified. It treats all classes equally by default.

**Enhanced Approach:**
Incorporates class weights $w_j = \frac{N}{|C_j|}$ to balance the influence of different classes during the tree construction.

## 2. Sampling with Replacement

**Conventional Random Forest:**
Uses bootstrap sampling (sampling with replacement) to create subsets of the training data for each tree, without considering class distribution.

**Enhanced Approach:**
Modifies the sampling process to ensure that each class is represented according to its weight. The probability of sampling an instance $(y_i = j)$ is 

$$
P(y_i = j) = \frac{w_j}{\sum_{k \in \{0, 1\}} w_k}
$$

## 3. Weighted Gini Impurity

**Conventional Random Forest:**
Uses standard Gini impurity for node splitting, calculated as

$$
G(t) = 1 - \sum_{j \in \{0, 1\}} p_j^2
$$

**Enhanced Approach:**
Uses weighted Gini impurity, incorporating class weights:

$$
G(t) = 1 - \sum_{j \in \{0, 1\}} \left(\frac{w_j p_j}{\sum_{k \in \{0, 1\}} w_k}\right)^2
$$

## 4. Voting Mechanism

**Conventional Random Forest:**
Each tree votes for a class, and the final prediction is made by majority vote.

**Enhanced Approach:**
Uses a weighted voting mechanism where votes are weighted by class weights: 

$$
y = \arg\max_j \sum_{m=1}^M w_j I(T_m(x) = j)
$$

where $I(T_m(x) = j)$ is an indicator function that is 1 if tree $T_m$ classifies $x$ as class $j$, and 0 otherwise.
