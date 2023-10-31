from data_loader import load_clean_data, load_noisy_data
from decision_tree import DecisionTree, Node

import matplotlib.pyplot as plt
import numpy as np

"""
    Node class annotated with attributes that will be used to calculate the correct
    coordinates when plotting the tree.
"""
class AnnotatedNode(Node):
    def __init__(self, attribute, value, left, right, x_coord, depth, shift, parent):
        super().__init__(attribute, value, left, right)
        self.x_coord = x_coord
        self.depth = depth
        self.shift = shift
        self.parent = parent

    def is_leftmost(self):
        return (not self.parent) or (self == self.parent.left) or (not self.parent.left)

"""
    Returns a new tree with the same structure as the input tree, but annotated with attributes that
    will be used to calculate the correct coordinates when plotting the tree.
"""
def annotate_tree(root, parent=None, depth=0):
    if not root:
        return None
    annotated_root = AnnotatedNode(root.attribute, root.value, left=None, right=None, x_coord=0, depth=depth, shift=0, parent=parent)
    annotated_root.left = annotate_tree(root.left, annotated_root, depth + 1)
    annotated_root.right = annotate_tree(root.right, annotated_root, depth + 1)
    return annotated_root

"""
    Returns a tree with x coordinates calculated to make better use of horizontal space,
    according to the Reingold-Tilford algorithm.

    Based on the algorithm described in https://rachel53461.wordpress.com/2014/04/20/algorithm-for-drawing-trees/,
    adapted in Python and simplified to work for binary trees only
"""
def calculate_coords(root):
    annotated_root = annotate_tree(root)
    calculate_initial_coords(annotated_root)
    calculate_final_coords(annotated_root, 0)
    return annotated_root

"""
    Calculate the x coordinate of the root based on its children, and accounting
    for any overlaps between subtrees.
"""
def calculate_initial_coords(root):
    if not root:
        return

    calculate_initial_coords(root.left)
    calculate_initial_coords(root.right)

    node_width = 2
    # Default x coordinate
    root.x_coord = 0
    # If the root has no children, either leave the x coordinate at 0, or shift
    # if the root is not the leftmost child of its parent
    if not root.left and not root.right:
        if not root.is_leftmost():
            root.x_coord = root.parent.left.x_coord + node_width
    elif not root.left or not root.right:
        # The root has at most one child, so place the root directly above it
        if root.is_leftmost():
            root.x_coord = root.left.x_coord if root.left else root.right.x_coord
        # If the root is not the leftmost child of its parent, shift it and its subtree accordingly
        else:
            root.x_coord = root.parent.left.x_coord + node_width
            root.shift = root.x_coord - root.left.x_coord if root.left else root.right.x_coord
    else:
        # The root has two children, so place it at the midpoint between them
        midpoint = (root.left.x_coord + root.right.x_coord) / 2
        if root.is_leftmost():
            root.x_coord = midpoint
        # If the root is not the leftmost child of its parent, shift it and its subtree accordingly
        else:
            root.x_coord = root.parent.left.x_coord + node_width
            root.shift = root.x_coord - midpoint

    # Check and fix any overlaps if the root is not a leaf
    if (root.left or root.right) and not root.is_leftmost():
        fix_overlaps(root, minimum_separation=int(1.5 * node_width))

"""
    Adjust shift values of overlapping subtrees.
"""
def fix_overlaps(root, minimum_separation):
    contour = {}
    calculate_contour(root, 0, contour, left=True)
    shift_value = 0.0
    sibling = root.parent.left if root.parent.left else root.parent.right

    if sibling and sibling != root:
        sibling_contour = {}
        calculate_contour(sibling, 0, sibling_contour, left=False)

        for depth in range(root.depth + 1, min(max(sibling_contour.keys()), max(contour.keys())) + 1):
            # At each depth of the contours, check if there is an overlap, and adjust shift
            # Subtrees overlap if the separation between the contours is less than the
            # minimum separation
            separation = contour[depth] - sibling_contour[depth]
            shift_value = max(shift_value, minimum_separation - separation)

    # Apply the shift if necessary
    if shift_value > 0:
        root.x_coord += shift_value
        root.shift += shift_value

"""
    Calculates the left/right contour of the tree starting at root.

    The contour is a set of x coordinates of the left/rightmost node at each depth.
"""
def calculate_contour(root, shift_sum, contour, left):
    if not root:
        return
    if not contour.get(root.depth):
        contour[root.depth] = (1 if left else -1) * np.inf
    # If we want the left contour, we take the smallest x coordinate on the current depth,
    # whereas for the right contour, we take the largest x coordinate
    contour[root.depth] = (min if left else max)(contour[root.depth], root.x_coord + shift_sum)

    shift_sum += root.shift
    calculate_contour(root.left, shift_sum, contour, left)
    calculate_contour(root.right, shift_sum, contour, left)

"""
    Calculates the final x coordinates of each node in the tree after the shift is applied.
"""
def calculate_final_coords(root, shift_sum):
    if not root:
        return
    root.x_coord += shift_sum
    shift_sum += root.shift
    calculate_final_coords(root.left, shift_sum)
    calculate_final_coords(root.right, shift_sum)

"""
    Plot the tree according to the Reingold-Tilford algorithm.
"""
def plot_tree(root):
    plot_tree_rt(calculate_coords(root))

def plot_tree_rt(root, depth=0):
    if not root:
        return

    # Display attribute and value in a text box
    text = ""
    if root.attribute == "leaf":
        text = "leaf: " + str(root.value)
    else:
        text = "X" + str(root.attribute) + " < " + str(root.value)

    plt.text(root.x_coord, depth, text, fontsize=6, verticalalignment='center', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue'))

    # Draw the lines between nodes in the tree
    points = []
    # If root.left exists, add a point in its position so that a line is drawn
    if root.left:
        points.append((root.left.x_coord, depth - 1))
    points.append((root.x_coord, depth))
    # If root.right exists, add a point in its position so that a line is drawn
    if root.right:
        points.append((root.right.x_coord, depth - 1))

    plt.plot(*zip(*points))

    # Plot subtrees
    plot_tree_rt(root.left, depth - 1)
    plot_tree_rt(root.right, depth - 1)

if __name__ == "__main__":
    # Check tree visualisation with the clean data
    clean_data = load_clean_data()
    clean_data_train = clean_data[:int(len(clean_data))]

    tree = DecisionTree()

    tree.train(clean_data_train)

    plot_tree(tree.root)
    plt.axis('off')
    plt.show()
