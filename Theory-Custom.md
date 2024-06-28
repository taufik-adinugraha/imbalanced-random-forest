# Handling Imbalanced Data within Random Forest Algorithm

The approach involves modifying the sampling process and adjusting the voting mechanism to account for class imbalances.

## Mathematical Formulation

### 1. Class Weights
Given a dataset $D = \{(x_i, y_i)\}_{i=1}^N$ where $x_i$ is a feature vector and $y_i \in \{0, 1\}$ is the class label, we can define class weights as:

$$
w_j = \frac{N}{|C_j|}
$$

where $|C_j|$ is the number of instances in class $j$. For binary classification with classes 0 and 1, the weights $w_0$ and $w_1$ can be used to balance the classes during the tree construction.

### 2. Sampling with Replacement
Each tree $T_m$ in the Random Forest is trained on a bootstrap sample. For imbalanced datasets, we can modify the sampling process to ensure that each class is represented according to its weight.

$$
P(y_i = j) = \frac{w_j}{\sum_{k \in \{0, 1\}} w_k}
$$

### 3. Weighted Gini Impurity
To split a node in the decision tree, we use a weighted Gini impurity. The Gini impurity for a node $t$ with class distribution $p_0$ and $p_1$ is given by:

$$
G(t) = 1 - \sum_{j \in \{0, 1\}} \left(\frac{w_j p_j}{\sum_{k \in \{0, 1\}} w_k}\right)^2
$$

### 4. Voting Mechanism
For the final classification, each tree $T_m$ votes for a class. The votes are weighted by the class weights:

$$
y = \arg\max_j \sum_{m=1}^M w_j I(T_m(x) = j)
$$

where $I(T_m(x) = j)$ is an indicator function that is 1 if tree $T_m$ classifies $x$ as class $j$, and 0 otherwise.
