class Node:
    def __init__(self, attribute, value, left, right):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
    

def same_label(data):
    return len(set(data[:, -1])) == 1

# TODO: Implement find_split functions.
def decision_tree_learning(train_data, depth):
    if same_label(train_data):
        return (Node("leaf", train_data[0, -1], None, None), depth)
    
    (split_attribute, split_value, left_data, right_data) = find_split(train_data)
    new_node = Node(split_attribute, split_value, None, None)

    (new_node.left, left_depth) = decision_tree_learning(left_data, depth+1)
    (new_node.right, right_depth) = decision_tree_learning(right_data, depth+1)

    return (new_node, max(left_depth, right_depth))
