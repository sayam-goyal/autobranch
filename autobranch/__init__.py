"""Autobranch: a generalized RL framework on Modal.

Hand it an environment bundle and a sentence of intent; it returns trained
weights. In between, a tree of agents in Modal Sandboxes searches reward
space, branching with git-worktree semantics and pruning lines that stop
paying off.
"""

__version__ = "0.1.0"
