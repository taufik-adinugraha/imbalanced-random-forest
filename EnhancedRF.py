import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

class EnhancedRandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """
        Initialize the Enhanced Random Forest with the specified parameters.
        
        Parameters:
        - n_estimators: Number of trees in the forest.
        - max_depth: Maximum depth of each tree.
        - min_samples_split: Minimum number of samples required to split an internal node.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.class_weights = None

    def fit(self, X, y):
        """
        Fit the Enhanced Random Forest model to the training data.
        
        Parameters:
        - X: Training features.
        - y: Training labels.
        """
        self.class_weights = self.compute_class_weights(y)
        self.trees = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            
            # Train a decision tree with sample weights
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            sample_weights = self.compute_sample_weights(y_sample)
            tree.fit(X_sample, y_sample, sample_weight=sample_weights)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict class labels for the input features.
        
        Parameters:
        - X: Input features.
        
        Returns:
        - Predicted class labels.
        """
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        weighted_votes = self.weighted_voting(tree_predictions)
        return np.argmax(weighted_votes, axis=1)

    def compute_class_weights(self, y):
        """
        Compute class weights to handle imbalanced data.
        
        Parameters:
        - y: Training labels.
        
        Returns:
        - Class weights dictionary.
        """
        class_counts = Counter(y)
        total_samples = len(y)
        weights = {cls: total_samples / count for cls, count in class_counts.items()}
        return weights

    def compute_sample_weights(self, y):
        """
        Compute sample weights based on class weights.
        
        Parameters:
        - y: Labels of the bootstrap sample.
        
        Returns:
        - Array of sample weights.
        """
        return np.array([self.class_weights[cls] for cls in y])

    def bootstrap_sample(self, X, y):
        """
        Generate a bootstrap sample with class weighting.
        
        Parameters:
        - X: Training features.
        - y: Training labels.
        
        Returns:
        - Bootstrap sample features and labels.
        """
        sample_indices = []
        for cls, weight in self.class_weights.items():
            cls_indices = np.where(y == cls)[0]
            sample_indices.extend(np.random.choice(cls_indices, int(weight * len(cls_indices)), replace=True))
        sample_indices = np.array(sample_indices)
        return X[sample_indices], y[sample_indices]

    def weighted_voting(self, tree_predictions):
        """
        Perform weighted voting on the predictions of each tree.
        
        Parameters:
        - tree_predictions: Predictions from all trees.
        
        Returns:
        - Weighted votes for each class.
        """
        weighted_votes = np.zeros((tree_predictions.shape[1], len(self.class_weights)))
        for cls, weight in self.class_weights.items():
            weighted_votes[:, cls] = np.sum(tree_predictions == cls, axis=0) * weight
        return weighted_votes
