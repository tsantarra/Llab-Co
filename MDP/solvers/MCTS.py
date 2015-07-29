from MDP.graph.State import State
from MDP.tree.TreeNode import TreeNode
from MDP.managers.Scenario import Scenario

from random import choice
from math import sqrt, log


def traverseNodes(node, scenario):
    """
    UCT down to leaf node.
    """
    utility = 0
    while node.untried_actions == [] and len(node.children) != 0 and not scenario.end(node.state):
        # Get UCT action. Progress game state.
        action = node.UCTSelectAction()
        newState = scenario.transitionState(node.state.copy(), action)
        utility += scenario.getUtility(newState)

        # Determine which node to traverse to next. Use double progressive widening due to stochastic robber behavior.
        nextNode = None
        if action in node.children:
            # If the action has been explored before, look for a matching childNode.
            for childNode in node.children[action]:
                if newState == childNode.state:
                    nextNode = childNode

        # If nextNode is still None, the new state is an undiscovered state; 
        # add new leaf (childNode).
        if nextNode is None: 
            print('traverse new state')
            nextNode = node.AddChild(newState, action, scenario)

        # Progress to next node.
        node = nextNode  
    return (utility, node)

def expandLeaf(node, scenario):
    """
    Expands a new node from current leaf node.
    """
    utility = 0

    # If conditions are met, add a leaf node.
    # Note: last condition covers indirectly adding a leaf through traverseNodes
    if node.untried_actions != [] and not scenario.end(node.state) and node.visits >= 1:  
        # Randomly select move. Progress game state.
        action = choice(node.untried_actions)
        newState = scenario.transitionState(node.state.copy(), action)
        utility = scenario.getUtility(newState)
            
        # Add new node to tree.
        node = node.AddChild(newState, action, scenario)
    return (utility, node)

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

def mcts(state, scenario, iterations, rootNode=None):
    """
    Search game tree according to MCTS. 
    """
    # If a rootNode is not specified, initialize a new one.
    if not rootNode:
        rootNode = MCTSNode(state, actionList=scenario.getActions(state), is_root=True)
    passes = iterations - rootNode.visits

    for step in range(passes):
        # Start at root
        node = rootNode
        utility = 0

        # UCT through existing nodes.
        (value, node) = traverseNodes(node, scenario)
        utility += value
        
        # Expand a new node from leaf.
        (value, node) = expandLeaf(node, scenario)
        utility += value

        # Simulate rest of game randomly.
        utility += rollout(node, scenario)
        
        # Backpropagate score.
        backpropagate(node, utility)

    return (rootNode.UCTSelectAction(), rootNode)



class MCTSNode(TreeNode):
    """
    A node in the game tree. 
    """
    def __init__(self, state, action = None, actionList = [], parent = None, is_root = False):
        """
        Initializes tree node with relevant information.
        """
        self.state = state                       # Current game state (clone of game instance). 
        self.action= action                      # The move that got us to this node - "None" for the root node.
        self.untried_actions = actionList        # Yet unexplored actions
        self.parent = parent                     # Parent node to this node
        self.is_root = is_root                   # Is this node the root node of the tree?
        
        self.children = {}                       # Action: childNode dictionary to keep links to children (through actions)
        self.value = 0.0                         # Average score of all paths through this node.
        self.visits = 0                          # Number of times this node has been visited.
        
    def UCTSelectAction(self, exploreFactor = 1):
        """
        Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
        lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
        exploration versus exploitation.
        """
        moveRewardMap = {}
        
        for child in [childNode for action, childrenFromAction in self.children.items() for childNode in childrenFromAction]:
                if child.action in moveRewardMap:
                    prevVisits = moveRewardMap[child.action][1]
                    newVal = (prevVisits*moveRewardMap[child.action][0] + child.visits*child.value)/(prevVisits+child.visits)
                    moveRewardMap[child.action] = [newVal, prevVisits+child.visits]
                else:
                    moveRewardMap[child.action] = [child.value, child.visits]

        best_move = sorted(moveRewardMap, key = lambda c: (moveRewardMap[c][0] + exploreFactor*sqrt(2*log(self.visits)/ moveRewardMap[c][1])))[-1]
        best_val = moveRewardMap[best_move][0]

        ties = [i for i in moveRewardMap.keys() if moveRewardMap[i][0] == best_val]

        move = choice(ties)
         
        return move 

    def AddChild(self, state, action, scenario):
        """
        Remove action from untried_actions and add a new child node for this move.
        Return the added child node.
        """
        node = MCTSNode(state, action, scenario.getActions(state), parent = self)
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        if action in self.children:
            self.children[action] += [node]
        else:
            self.children[action] = [node]
        
        return node
    
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
        return "[" + str(self.action) + " Val:" + str(self.value) + " Vis:" + str(self.visits) + "]" 
     
    def TreeToString(self, horizon = 1, indent = 0):
        """
        Builds a string representation of the tree via recursive tree traversal.
        """
        string = ''.join(['| ' for i in range(indent)]) + str(self) + '\n'
        if horizon > 0:
            for child in [node for children in self.children.values() for node in children]:
                 string += child.TreeToString(horizon-1, indent+1)
        return string
