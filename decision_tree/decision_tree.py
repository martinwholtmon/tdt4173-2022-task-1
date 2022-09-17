from dataclasses import replace
from distutils import dep_util
import numpy as np
import pandas as pd
import copy

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class Tree:
    def __init__(self, label=""):
        self.nodes = []
        self.label = label
        self.most_common_value = ""

    def set_label(self, label):
        self.label = label

    def print(self, space=""):
        print(f"{space}label = {self.label}")
        for node in self.nodes:
            node.print(space + "| ")


class DecisionTree:
    def __init__(
        self,
        target_value_success="Yes",
        max_depth=float("inf"),
        min_samples_split=0,
        min_samples_leaf=0,
        variance_threshold=1,
    ):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self._tree = Tree()
        self._target_values = []
        self._ground_thruth = []
        self._target_value_success = target_value_success
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._variance_threshold = variance_threshold

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

        # Store the target_values for failure and success
        all_target_values = list(dict.fromkeys(y))
        failure = ""
        success = self._target_value_success

        if all_target_values[0] == success:
            failure = all_target_values[1]
        else:
            failure = all_target_values[0]

        self._target_values = [
            failure,
            success,
        ]
        self._ground_thruth = y

        # convert ground-truth to 0 1, success = 1
        y = np.array(pd.Series(np.where(y == self._target_value_success, 1, 0), y))

        # Preprossessing: Feature Selection
        remove_col = []
        for i in range(attributes.shape[0]):
            # Get all occurances of attribute
            occurances, _ = get_occurances(X, y, i)
            # if only two values, i.e. boolean yes/no
            if len(occurances) == 2:
                # Caluclate occurance of each value
                sum_occ = list(map(sum, occurances.values()))

                # Calulcate probability of occurance
                p = np.min(sum_occ) / np.max(sum_occ)

                # Remove attribute if the probability of occurance
                # is higher than variance_threshold
                if p > self._variance_threshold:
                    remove_col.append(i)
        X = np.delete(X, remove_col, 1)

        # id3
        self._tree = id3(
            X,
            y,
            attributes,
            self._target_values,
            0,
            self._max_depth,
            self._min_samples_split,
            self._min_samples_leaf,
        )
        # self._tree.print("")

    def post_prune(self, X, y):
        # Post pruning
        # for each rule
        #   acc = accuracy(y, predict(tree))
        #
        #   for each precondition
        #       old_tree = copy.deepcopy(self._tree)
        #       update tree
        #       calculate accuracy
        #       if new_acc < old_acc:
        #           self._tree = old_tree
        rules = self.get_rules()

        for rule in rules:
            acc = accuracy(y, self.predict(X))
            # Prune the tree from bottom up
            for pre_con, _ in list(reversed(rule[0])):
                old_tree = copy.deepcopy(self._tree)

                # Modify tree
                prune_node(self._tree, rule[0], pre_con, Tree(rule[1]))

                # Calulate accuracy
                new_acc = accuracy(y, self.predict(X))

                if new_acc <= acc:
                    self._tree = old_tree

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
            results.append(
                get_prediction(tree, attributes, list(e), self._target_values)
            )
        # print(results)
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
        rules = []
        get_leaf_nodes(self._tree, [], rules, self._target_values)
        return rules


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


def id3(
    examples,
    target_attribute,
    attributes,
    target_values,
    depth,
    max_depth,
    min_samples_split,
    min_samples_leaf,
) -> Tree:
    n_examples, n_attributes = examples.shape
    root = Tree()
    # Calculate S
    S = np.array([0, 0])
    for i in range(n_examples):
        if target_attribute[i] == 1:
            S[0] += 1
        else:
            S[1] += 1

    # Most common value
    if np.argmin(S) == 0:
        most_common_value = target_values[1]
    else:
        most_common_value = target_values[0]
    root.most_common_value = most_common_value

    # All positive
    if S[0] == n_examples:
        root.set_label(target_values[1])
        return root

    # All negative
    if S[1] == n_examples:
        root.set_label(target_values[0])
        return root

    # Attributes is empty
    if n_attributes == 0 or depth == max_depth or n_examples <= min_samples_leaf:
        root.set_label(most_common_value)
        return root

    # Select best attribute
    entropies = []
    all_attributes = []
    for i in range(n_attributes):
        # get occurances of each sub-attribute
        arr, total = get_occurances(examples, target_attribute, i)

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
        # source: https://stackoverflow.com/a/60366885
        rows_to_keep = np.argwhere(np.any(examples == attribute, axis=1))[:, 0]
        examples_new = examples[rows_to_keep]
        target_attribute_new = target_attribute[rows_to_keep]
        examples_new = np.delete(examples_new, selected_attribute, 1)

        if examples_new.size <= min_samples_split:
            node.nodes = [Tree(most_common_value)]
        else:
            node.nodes = [
                id3(
                    examples_new,
                    target_attribute_new,
                    np.delete(attributes, selected_attribute),
                    target_values,
                    depth + 1,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                )
            ]
        nodes.append(node)

    # Update root
    root.label = attributes[selected_attribute]
    root.nodes = nodes
    return root


def get_occurances(examples, target_attribute, attribute_pos):
    # get occurances of each sub-attribute
    arr = {}
    total = 0
    for example in range(examples.shape[0]):
        key = examples[example][attribute_pos]
        if key not in arr:
            arr[key] = [0, 0]
        if target_attribute[example] == 1:
            arr[key] = [arr[key][0] + 1, arr[key][1]]
        else:
            arr[key] = [arr[key][0], arr[key][1] + 1]
        total += 1
    return arr, total


def get_prediction(node, attributes, example, target_values) -> str:
    attribute = node.label
    if attribute in target_values:
        return node.label

    # Select next node / correct label
    next_node = Tree()
    for i in range(len(node.nodes)):
        if node.nodes[i].label in example:
            next_node = node.nodes[i]

    # Path does not exist
    if next_node.label == "":
        return node.most_common_value

    # Print
    # print(attributes)
    # print(example)
    # print(f"attribute = {attribute}")
    # print(f"Selected example = {next_node.label}")
    # print(f"Next label = {next_node.nodes[0].label}")

    # Remove attribute from list and update example
    attributes = np.delete(attributes, np.where(attributes == attribute))
    example.remove(next_node.label)

    # Continue
    return get_prediction(next_node.nodes[0], attributes, example, target_values)


def get_leaf_nodes(node, path, rules, target_values):
    """Will create the rules for the decision tree in the form:
    [[[['Outlook', 'Rain'], ['Wind', 'Strong']], 'No'], ...]

    Args:
        node (Tree): the nodes of the tree
        path (list): the current path
        rules (list): the rules
        target_values (list): the values defining success/failure
    """
    # Check if leaf node
    if node.label in target_values:
        # Store a COPY (no pointers!!) of the path with the label
        rules.append([path[:], str(node.label)])
    else:
        # Go to next node
        for n in node.nodes:
            # Add to path
            path.append([node.label, n.label])
            get_leaf_nodes(n.nodes[0], path, rules, target_values)
    if len(path) != 0:
        path.pop()


def prune_node(node, path, parent, replacement):
    if node.label == parent:
        node.nodes = [replacement]
    else:
        for n in node.nodes:
            if n.label == path[0][1]:
                prune_node(n.nodes[0], path[1:], parent, replacement)
