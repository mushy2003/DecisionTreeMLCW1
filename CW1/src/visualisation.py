from data_loader import clean_data
from decision_tree import DecisionTree

import matplotlib.pyplot as plt

def plot_tree(root, x_coord, depth):
    # TODO: Fix issue where text boxes overlap if the tree is too large
    if not root:
        return

    # Display attribute and value in a text box
    text = ""
    if root.left or root.right:
        text = "X" + str(root.attribute) + " < " + str(root.value)
    else:
        text = "leaf: " + str(root.value)

    plt.text(x_coord, depth, text, verticalalignment='center', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue'))

    # Draw the lines between nodes in the tree
    max_width = 2 ** depth
    points = []
    # If root.left exists, add a point in its position so that a line is drawn
    if root.left:
        points.append((x_coord - max_width, depth - 1))
    points.append((x_coord, depth))
    # If root.right exists, add a point in its position so that a line is drawn
    if root.right:
        points.append((x_coord + max_width, depth - 1))

    plt.plot(*zip(*points))

    # Plot subtrees
    plot_tree(root.left, x_coord - max_width, depth - 1)
    plot_tree(root.right, x_coord + max_width, depth - 1)

if __name__ == "__main__":
    # Check tree visualisation with the clean data
    clean_data_train = clean_data[:int(0.8 * len(clean_data))]

    tree = DecisionTree()

    tree.train(clean_data_train)

    plot_tree(tree.root, 0, tree.depth)
    plt.axis('off')
    plt.show()
