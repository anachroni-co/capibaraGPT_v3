# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for working with tree-like container data structures.

This module provides a small set of utility functions for working with tree-like
data structures, such as nested tuples, lists, and dicts. We call these
structures pytrees. They are trees in that they are defined recursively (any
non-pytree is a pytree, i.e. a leaf, and any pytree of pytrees is a pytree) and
can be operated on recursively (object identity equivalence is not preserved by
mapping operations, and these structures cannot contain reference cycles).

The set of Python types that are considered pytree nodes (e.g. that can be
mapped over, rather than treated as leaves) is extensible. There is a single
module-level registry of types, and class hierarchy is ignored. By registering a
new pytree node type, that type in effect becomes transparent to these utility
functions in this file.

The primary purpose of this module is to enable the interoperability between
your defined data structures and JAX transformations (e.g. `jit`). This is not
meant to be a general purpose tree-like data structure handling library.

See the `JAX pytrees note <pytrees.html>`_
for examples.
"""

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

"""
JAX tree_util - Minimal implementation

Utilities for pytree handling.
"""

try:
    # Try to use real JAX if available
    from jax import tree_util as real_tree_util
    tree_flatten = real_tree_util.tree_flatten
    tree_unflatten = real_tree_util.tree_unflatten
    tree_map = real_tree_util.tree_map
    tree_leaves = real_tree_util.tree_leaves

except ImportError:
    # Simple fallback implementation
    def tree_flatten(tree):
        """Flatten a pytree."""
        if isinstance(tree, (list, tuple)):
            flat = []
            for item in tree:
                sub_flat, _ = tree_flatten(item)
                flat.extend(sub_flat)
            return flat, None
        else:
            return [tree], None

    def tree_unflatten(tree_def, flat_tree):
        """Unflatten a pytree."""
        return flat_tree

    def tree_map(f, tree):
        """Map function over pytree."""
        if isinstance(tree, (list, tuple)):
            return type(tree)(tree_map(f, item) for item in tree)
        else:
            return f(tree)

    def tree_leaves(tree):
        """Get leaves of pytree."""
        flat, _ = tree_flatten(tree)
        return flat

__all__ = ['tree_flatten', 'tree_unflatten', 'tree_map', 'tree_leaves']
