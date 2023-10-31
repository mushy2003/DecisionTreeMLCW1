import numpy as np

class Node:
    def __init__(self, attribute, value, left, right):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
    
class DecisionTree:
    def __init__(self):
        self.root = None
        self.depth = 0
    
    """
    Trains/Creates the decision tree using the training data. Internally uses the _decision_tree_learning method.
    """
    def train(self, train_data):
        (self.root, self.depth) = self._decision_tree_learning(train_data, 0)
    
    """
    Returns the predictions of the decision tree given test data.
    """
    def predict(self, test_data):
        # Note test data will have one less column than training data since it won't have labels
        y_preds = np.zeros(len(test_data))

        for row in range(len(test_data)):
            curr_node = self.root
            while curr_node.attribute != "leaf":
                if test_data[row, curr_node.attribute] <= curr_node.value:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right
            y_preds[row] = curr_node.value
        
        return y_preds

    """
    Tries to find the best split in the training data, the split that leads to the highest information gain is considered the best split.
    Returns the best split attribute, value and the data remaining on either side of the split.
    """
    def _find_split(self, data):
        _, num_attributes = data.shape
        num_attributes -= 1 # Since last column is the label
        best_info_gain = 0
        best_split_attribute = None
        best_split_value = None
        left_data = None
        right_data = None

        def entropy(dataset):
            _, counts = np.unique(dataset[:, -1], return_counts=True)
            total = np.sum(counts)
            p_ks = counts / total
            entropy = -1 * np.sum(p_ks * np.log2(p_ks))
            
            return entropy

        def information_gain(data, curr_left_data, curr_right_data):
            remainder = (len(curr_left_data) / len(data)) * entropy(curr_left_data) + (len(curr_right_data) / len(data)) * entropy(curr_right_data)
            return entropy(data) - remainder

        # Iterates through each attribute, setting the split value as a midpoint between attribute values. 
        # Selects the attribute and corresponding split value that leads to the highest information gain. 
        for attribute in range(num_attributes):
            sorted_values = np.sort(data[:, attribute])
            for i in range(1, len(sorted_values)):
                curr_split_value = (sorted_values[i-1] + sorted_values[i]) / 2
                curr_left_data = data[data[:, attribute] <= curr_split_value]
                curr_right_data = data[data[:, attribute] > curr_split_value]

                if len(curr_left_data) == 0 or len(curr_right_data) == 0:
                    continue

                curr_info_gain = information_gain(data, curr_left_data, curr_right_data)

                if curr_info_gain > best_info_gain:
                    best_info_gain = curr_info_gain
                    best_split_attribute, best_split_value = attribute, curr_split_value
                    left_data, right_data = curr_left_data, curr_right_data
        
        return best_split_attribute, best_split_value, left_data, right_data


    """
    Responsible for creating the decision tree itself given the training data.
    """
    def _decision_tree_learning(self, train_data, depth):
        
        def same_label(data):
            return len(set(data[:, -1])) == 1
        
        if same_label(train_data):
            return (Node("leaf", train_data[0, -1], None, None), depth)
        
        (split_attribute, split_value, left_data, right_data) = self._find_split(train_data)
        new_node = Node(split_attribute, split_value, None, None)

        (new_node.left, left_depth) = self._decision_tree_learning(left_data, depth+1)
        (new_node.right, right_depth) = self._decision_tree_learning(right_data, depth+1)

        return (new_node, max(left_depth, right_depth))
