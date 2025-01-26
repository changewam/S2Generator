# -*- coding: utf-8 -*-
"""
Created on 2025/01/23 18:25:07
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from numpy import ndarray
import scipy.special
from typing import Optional, Union, List

from S2Generator.params import Params

operators_real = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "abs": 1,
    "inv": 1,
    "sqrt": 1,
    "log": 1,
    "exp": 1,
    "sin": 1,
    "arcsin": 1,
    "cos": 1,
    "arccos": 1,
    "tan": 1,
    "arctan": 1,
    "pow2": 1,
    "pow3": 1,
}

operators_extra = {"pow": 2}

math_constants = ["e", "pi", "euler_gamma", "CONSTANT"]
all_operators = {**operators_real, **operators_extra}

SPECIAL_WORDS = [
    "<EOS>",
    "<X>",
    "</X>",
    "<Y>",
    "</Y>",
    "</POINTS>",
    "<INPUT_PAD>",
    "<OUTPUT_PAD>",
    "<PAD>",
    "(",
    ")",
    "SPECIAL",
    "OOD_unary_op",
    "OOD_binary_op",
    "OOD_constant",
]


class Node(object):
    """Generate a node in the sampling tree"""

    def __init__(self, value: Union[str, int], params: Params, children: list = None) -> None:
        # The specific value stored in the current node
        self.value = value
        # The list of child nodes that the current node points to
        self.children = children if children else []
        self.params = params

    def push_child(self, child: "Node") -> None:
        """Add a child node to the current node"""
        self.children.append(child)

    def prefix(self) -> str:
        """Get all the contents of this tree using a recursive traversal starting from the current root node"""
        s = str(self.value)
        for c in self.children:
            s += "," + c.prefix()
        return s

    def qtree_prefix(self) -> str:
        """Get all the contents of this tree using a recursive traversal starting from the current root node, storing the result in a list"""
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self) -> str:
        """Output the entire symbolic expression using in-order traversal"""
        nb_children = len(self.children)  # Get the number of children
        if nb_children == 0:
            # If there are no children, the current node is a leaf node
            if self.value.lstrip("-").isdigit():
                return str(self.value)
            else:
                s = str(self.value)
                return s  # Output the content of the leaf node
        if nb_children == 1:
            # If there is only one child, it indicates a unary operator
            s = str(self.value)
            # Handle different types of unary operators
            if s == "pow2":
                s = "(" + self.children[0].infix() + ")**2"
            elif s == "pow3":
                s = "(" + self.children[0].infix() + ")**3"
            else:
                # Output in the form of f(x), where f is functions like sin and cos
                s = s + "(" + self.children[0].infix() + ")"
            return s
        # If the current node is a binary operator, combine using the intermediate terms
        s = "(" + self.children[0].infix()
        for c in self.children[1:]:
            s = s + " " + str(self.value) + " " + c.infix()
        return s + ")"

    def val(self, x: ndarray, deterministic: Optional[bool] = True) -> ndarray:
        """Evaluate the symbolic expression using specific numerical sequences"""
        if len(self.children) == 0:
            # If the node is a leaf node, it is a symbolic variable or a random constant
            if str(self.value).startswith("x_"):
                # Handle symbolic expressions
                _, dim = self.value.split("_")
                dim = int(dim)
                return x[:, dim]
            elif str(self.value) == "rand":
                # Handle random constants
                if deterministic:
                    return np.zeros((x.shape[0],))
                return np.random.randn(x.shape[0])
            elif str(self.value) in math_constants:
                return getattr(np, str(self.value)) * np.ones((x.shape[0],))
            else:
                return float(self.value) * np.ones((x.shape[0],))

        # Handle various binary operators and perform specific calculations recursively
        if self.value == "add":
            return self.children[0].val(x) + self.children[1].val(x)  # Addition
        if self.value == "sub":
            return self.children[0].val(x) - self.children[1].val(x)  # Subtraction
        if self.value == "mul":
            m1, m2 = self.children[0].val(x), self.children[1].val(x)  # Multiplication
            # Handle exceptions in penalized calculations
            try:
                return m1 * m2
            except Exception as e:
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow":
            m1, m2 = self.children[0].val(x), self.children[1].val(x)  # Exponentiation
            try:
                return np.power(m1, m2)
            except Exception as e:
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "max":
            return np.maximum(self.children[0].val(x), self.children[1].val(x))  # Maximum
        if self.value == "min":
            return np.minimum(self.children[0].val(x), self.children[1].val(x))  # Minimum
        if self.value == "div":
            # Ensure denominator is not zero
            denominator = self.children[1].val(x)
            denominator[denominator == 0.0] = np.nan
            try:
                return self.children[0].val(x) / denominator  # Division
            except Exception as e:
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans

        # Handle various unary operators
        if self.value == "inv":
            # Ensure denominator is not zero
            denominator = self.children[0].val(x)
            denominator[denominator == 0.0] = np.nan
            try:
                return 1 / denominator  # Reciprocal
            except Exception as e:
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "log":
            numerator = self.children[0].val(x)
            # Ensure logarithm inputs are not negative or zero
            if self.params.use_abs:
                # Use log(abs(.)) if specified
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator <= 0.0] = np.nan
            try:
                return np.log(numerator)  # Logarithm
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "sqrt":
            numerator = self.children[0].val(x)
            # Ensure square root inputs are non-negative
            if self.params.use_abs:
                # Apply absolute value if specified
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator < 0.0] = np.nan
            try:
                return np.sqrt(numerator)  # Square root
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow2":
            numerator = self.children[0].val(x)
            try:
                return numerator ** 2  # Square
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow3":
            numerator = self.children[0].val(x)
            try:
                return numerator ** 3  # Cube
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "abs":
            return np.abs(self.children[0].val(x))  # Absolute value
        if self.value == "sign":
            return (self.children[0].val(x) >= 0) * 2.0 - 1.0  # Sign function
        if self.value == "step":
            x = self.children[0].val(x)  # Step function
            return x if x > 0 else 0
        if self.value == "id":
            return self.children[0].val(x)  # Identity mapping
        if self.value == "fresnel":
            return scipy.special.fresnel(self.children[0].val(x))[0]
        if self.value.startswith("eval"):
            n = self.value[-1]
            return getattr(scipy.special, self.value[:-1])(n, self.children[0].val(x))[0]
        else:
            fn = getattr(np, self.value, None)
            if fn is not None:
                try:
                    return fn(self.children[0].val(x))
                except Exception as e:
                    nans = np.empty((x.shape[0],))
                    nans[:] = np.nan
                    return nans
            fn = getattr(scipy.special, self.value, None)
            if fn is not None:
                return fn(self.children[0].val(x))
            assert False, "Could not find function"

    def get_recurrence_degree(self) -> int:
        """Get the maximum variable index for leaf nodes when the current node is the root"""
        recurrence_degree = 0
        if len(self.children) == 0:
            # If the current node is a leaf node
            if str(self.value).startswith("x_"):
                _, offset = self.value.split("_")
                offset = int(offset)
                if offset > recurrence_degree:
                    recurrence_degree = offset
            return recurrence_degree
        return max([child.get_recurrence_degree() for child in self.children])

    def replace_node_value(self, old_value: str, new_value: str) -> None:
        """Traverse the entire symbolic expression and replace it with a specific value"""
        if self.value == old_value:
            self.value = new_value
        for child in self.children:
            child.replace_node_value(old_value, new_value)

    def __len__(self) -> int:
        """Output the total length of the expression with the current node as the root node"""
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self) -> str:
        # infix a default print
        return self.infix()

    def __repr__(self) -> str:
        # infix a default print
        return str(self)


class NodeList(object):
    """A list that stores the entire multivariate symbolic expression"""

    def __init__(self, nodes: List[Node]) -> None:
        self.nodes = []  # Initialize the list to store root nodes
        for node in nodes:
            self.nodes.append(node)
        self.params = nodes[0].params

    def infix(self) -> str:
        """Connect all multivariate symbolic expressions with |"""
        return " | ".join([node.infix() for node in self.nodes])  # In-order traversal of the tree

    def prefix(self) -> str:
        """Connect all multivariate symbolic expressions with ,|,"""
        return ",|,".join([node.prefix() for node in self.nodes])

    def val(self, xs: ndarray, deterministic: Optional[bool] = True) -> ndarray:
        """Sample the entire multivariate symbolic expression to obtain a specific numerical sequence"""
        batch_vals = [np.expand_dims(node.val(np.copy(xs), deterministic=deterministic), -1) for node in self.nodes]
        return np.concatenate(batch_vals, -1)

    def replace_node_value(self, old_value: str, new_value: str) -> None:
        """Traverse the entire symbolic expression to replace a specific value"""
        for node in self.nodes:
            node.replace_node_value(old_value, new_value)

    def __len__(self) -> int:
        # Get the length of the entire multivariate symbolic expression
        return sum([len(node) for node in self.nodes])

    def __str__(self) -> str:
        """Output the multivariate symbolic expression in string form"""
        return self.infix()

    def __repr__(self) -> str:
        return str(self)
