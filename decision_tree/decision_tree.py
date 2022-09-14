from cProfile import label
from cgi import test
import re
import numpy as np
import pandas as pd

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class Tree:
    def __init__(self, label=""):
        self.nodes = []
        self.label = label

    def set_label(self, label):
        self.label = label

    def print(self):
        print(f"label = {self.label}")
        for node in self.nodes:
            node.print()
            print("up")


class DecisionTree:
    def __init__(self):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self._tree = Tree()

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
        X = np.asarray(X)
        y = np.asarray(y)

        # id3
        tree = id3(X, y, attributes)
        self._tree = tree
        # tree.print()

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
        attributes = np.asarray(X.columns)
        X = np.asarray(X)
        tree = self._tree

        results = []
        for e in X:
            results.append(get_leaf_node(tree, attributes, e))
        return np.asarray(results)

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
        # raise NotImplementedError()


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


def id3(examples, target_attribute, attributes):
    n_examples, n_attributes = examples.shape
    root = Tree()
    # Calculate S
    S = np.array([0, 0])
    for i in range(n_examples):
        if target_attribute[i] == "Yes":
            S[0] += 1
        else:
            S[1] += 1

    # Most common value
    if np.argmin(S) == 0:
        most_common_value = "Yes"
    else:
        most_common_value = "No"

    # All positive
    if S[0] == n_examples:
        root.set_label("Yes")
        return root

    # All negative
    if S[1] == n_examples:
        root.set_label("No")
        return root

    # Attributes is empty
    if len(attributes) == 0:
        root.set_label(most_common_value)
        return root

    # Select best attribute
    entropies = []
    all_attributes = []
    for i in range(n_attributes):
        # get occurances of each sub-attribute
        arr = {}
        total = 0
        for example in range(n_examples):
            key = examples[example][i]
            if key not in arr:
                arr[key] = [0, 0]
            if target_attribute[example] == "Yes":
                arr[key] = [arr[key][0] + 1, arr[key][1]]
            else:
                arr[key] = [arr[key][0], arr[key][1] + 1]
            total += 1

        # calculate entropy for each attribute
        ent = entropy(S)
        for _, v in arr.items():
            v = np.asarray(v)
            ent = ent - sum(v) / total * entropy(v)
        entropies.append(ent)
        all_attributes.append(list(arr.keys()))
        # print(f"Entrophy for attribute {attributes[i]} = {ent}")
    selected_attribute = np.argmax(entropies)

    # print(f"selected attribute: {attributes[selected_attribute]}")

    nodes = []
    for attribute in all_attributes[selected_attribute]:
        node = Tree(attribute)

        # Create subset
        # source: https://stackoverflow.com/a/60366885'
        rows_to_keep = np.argwhere(np.any(examples == attribute, axis=1))[:, 0]
        examples_new = examples[rows_to_keep]
        target_attribute_new = target_attribute[rows_to_keep]
        examples_new = np.delete(examples_new, selected_attribute, 1)

        if examples_new.size == 0:
            node.nodes = [Tree(most_common_value)]
        else:
            node.nodes = [
                id3(
                    examples_new,
                    target_attribute_new,
                    np.delete(attributes, selected_attribute),
                )
            ]
        nodes.append(node)

    # Update root
    root.label = attributes[selected_attribute]
    root.nodes = nodes
    return root


def get_leaf_node(node, attributes, example) -> str:
    if node.label == "Yes" or node.label == "No":
        return node.label

    # find attribute position
    a_pos = np.where(attributes == node.label)[0]
    if a_pos.size == 0:
        return "Err"
    else:
        a_pos = a_pos[0]

    # Select next node
    next_node = Tree()
    for e in example[a_pos:]:
        for n_node in node.nodes:
            if e == n_node.label:
                next_node = n_node.nodes[0]
    return get_leaf_node(next_node, attributes, example)
