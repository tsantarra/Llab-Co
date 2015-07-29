"""
Experimental solver. Ideas:
    - MCTS slow for simulating so much; (finite horizon) Value Iteration is fast for not doing so (also, dynamic programming).
    - MCTS has the benefit of asymmetric tree search.
    - Can we combine the two?
        + Asymmetric tree growth
        + Optional rollouts
        + Dynamic programming (?)
        + Start from leaf (modify UCT for set of leaves); expand one step (all children)
            -> UCT = Expected value / depth + exploration factor

Goals:
    - Expected time
        + MCTS
            -> Faster execution
            -> Same(?) nodes expanded (with rollouts)
        + Value Iteration
            -> Same in worst case (finding singular, deep goal state)
    - Memory
        + Same as MCTS (without dynamic prog), as stochastic domains require state saving 
        + Same as VI in worst case (better in best case)

Possible related papers:
    Trial-based Heuristic Tree Search for Finite Horizon MDPs - Keller and Helmert

"""

from MDP.graph.State import State
from MDP.tree.TreeNode import TreeNode
from MDP.managers.Scenario import Scenario

from random import choice, uniform
from math import sqrt, log, e


def avi(state, scenario, iterations):
    """
    Search game tree according to AVI. 
    """
    # If a rootNode is not specified, initialize a new one.
    rootNode = AVINode(state)
    nodeTable = {}
    leafList = []

    for step in range(iterations):
        # Start at root
        node = rootNode
        utility = 0

        # Select a leaf
        (value, node) = chooseLeaf(leafList)
        utility += value
        
        # Expand all children from leaf.
        (value, node) = expandLeaf(node, scenario, leafList, nodeTable)
        utility += value

        # Simulate rest of game.
        #utility += rollout(node, scenario)
        
        # Update nodes to root.
        backpropagate(node, utility)

    return (rootNode.SelectAction(), rootNode)

def chooseLeaf(leafList):
    """
    Use modified UCT to select leaf:
        Avg val/depth + exploration factor * sqrt( 1/ visits)
    """
    #value is only immediate value for leaves
    #must factor in path value (through parent)
    #also, all leaves will only have a single visit
    #for now, we'll use a boltzmann distribution
    vals = {node: e**node.cumulative for node in leafList}
    total = sum(vals.values())
    target = uniform()
    running_total = 0
    for node, prob in vals.items():
        running_total += prob/total
        if running_total > target:
            return (node.cumulative, node)

    node, prob = vals.items()[-1]
    return (node.cumulative, node)

def expandLeaf(node, scenario, leafList, nodeTable):
    """
    Expands a new node from current leaf node.
    """
    # Remove leaf from leafList
    leafList.remove(node)

    # For every possible action
    for action in scenario.getActions(node.state):
        # For every possible resulting state
        for resultingState in scenario.transitionState(node.state, action).keys():

            #If the state at depth+1 has not yet been computed, add leaf.
            if (node.depth+1, resultingState) not in nodeTable:
                newLeaf = node.AddChild(resultingState, action, scenario)
                nodeTable[(newLeaf.depth, newLeaf.state)] = newLeaf
                leafList.append(newLeaf) 
            else:
                # Add existing node as child to current node.
                node.AddExistingChild(nodeTable[(newLeaf.depth, newLeaf.state)], action)
            

def rollout(node, scenario):
    """
    Plays remainder of game by minimizing manhattan distance.
    """
    utility = 0
    state = node.state.copy()

    while not scenario.end(state):
        action = choice(scenario.getActions(state))
        state = scenario.transitionState(state, action)
        utility += scenario.getUtility(state)

    return utility

def backpropagate(node, utility):
    """
    Updates tree along path according to reward.
    """
    while node != None:
        node.Update(utility)
        node = node.parent





class AVINode(TreeNode):
    """
    A node in the game tree. 
    """
    def __init__(self, state, scenario, action = None, parents = [], depth = 0):
        """
        Initializes tree node with relevant information.
        """
        self.state = state                       # Current game state (clone of game instance). 
        self.action= action                      # The move that got us to this node - "None" for the root node.
        self.parents = parents                   # Parents of this node (multiple due to graph dynamic programming)
        self.depth = depth                       # Depth of node in tree, for normalization
        
        self.children = {}                       # Action: childNode dictionary to keep links to children (through actions)
        self.cumulative = max([0] + [parent.cumulative for parent in parents]) + scenario.getUtility(state) # Max value possible accumulated so far
        self.value = scenario.getUtility(state)  # Expectation of value through this node.
        self.visits = 0                          # Number of times this node has been visited.


    def AddChild(self, state, action, scenario):
        """
        Remove action from untried_actions and add a new child node for this move.
        Return the added child node.
        """
        node = AVINode(state, action, parents = [self], depth = self.depth + 1)
        node.value = scenario.getUtility(node.state) 
        if action in self.children:
            self.children[action] += [node]
        else:
            self.children[action] = [node]
        
        return node

    def AddExistingChild(self, node, action):
        """
        Adds existing child to child list.
        """
        if action in self.children:
            self.children[action] += [node]
        else:
            self.children[action] = [node]

    def Update(self, utility):
        """
        Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerTurn.
        """
        self.visits += 1
        self.value = (self.value*(self.visits-1) + utility)/self.visits
        
    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return "[" + str(self.action) + " Val:" + str(self.value)  + " Dep:" + str(self.depth) + " Vis:" + str(self.visits) + "]" 
     
    def TreeToString(self, horizon = 1, indent = 0):
        """
        Builds a string representation of the tree via recursive tree traversal.
        """
        string = ''.join(['| ' for i in range(indent)]) + str(self) + '\n'
        if horizon > 0:
            for child in [node for children in self.children.values() for node in children]:
                 string += child.TreeToString(horizon-1, indent+1)
        return string
