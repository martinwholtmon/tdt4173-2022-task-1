import numpy as np
import pandas as pd

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    def __init__(self):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        pass

    def fit(self, X, y):
        """
        Generates a decision tree for classification

        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        # Prepare arrays
        attributes = np.asarray(X.columns)
        print(attributes)
        X = np.asarray(X)
        y = np.asarray(y)
        n_examples, n_attributes = X.shape

        print(X)
        # print(y)
        positive = 0
        negative = 0
        for i in range(n_examples):
            if y[i] == "Yes":
                positive += 1
            else:
                negative += 1
        S = np.array([positive, negative])

        # id3
        root = ["test"]
        pos = 0
        curr_entropy = 0
        curr_pos = 0
        for attribute in range(n_attributes):
            # get occurances
            arr = {}
            total = 0
            for example in range(n_examples):
                key = X[example][attribute]
                if key not in arr:
                    arr[key] = [0, 0]
                if y[example] == "Yes":
                    arr[key] = [arr[key][0] + 1, arr[key][1]]
                else:
                    arr[key] = [arr[key][0], arr[key][1] + 1]
                total += 1

            # pick attribute
            ent = entropy(S)
            for _, v in arr.items():
                v = np.asarray(v)
                ent = ent - sum(v) / total * entropy(v)
            print(f"Entrophy for attribute {attributes[attribute]} = {ent}")

            if ent > curr_entropy:
                root[pos] = attributes[attribute]
                curr_entropy = ent
                curr_pos = attribute
        np.delete(attributes, curr_pos)
        pos += 1

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.

        Returns:
            A length m vector with predictions
        """
        # TODO: Implement
        raise NotImplementedError()

    def get_rules(self):
        """
        Returns the decision tree as a list of rules

        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label

            attr1=val1 ^ attr2=val2 ^ ... => label

        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        raise NotImplementedError()


# --- Some utility functions


def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy

    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning

    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0

    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.

    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))
