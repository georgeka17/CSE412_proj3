# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from os import stat
import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount # gamma
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0 #holds values of current state
     
    "*** YOUR CODE HERE ***"
    states = self.mdp.getStates()
    startState = self.mdp.getStartState()

    #initialize current state dictionary values to be all 0
    #while loop???
    for iteration in range(self.iterations):
        nextValues = util.Counter() # holds values of next_state
        for state in states: #
            value = float('-inf');
            actions = self.mdp.getPossibleActions(state);
            for action in actions: #where's the end of the grid ??
              statesAndProbs = (self.mdp.getTransitionStatesAndProbs(state, action))
              sum = 0
              for stateAndProb in statesAndProbs:
                  sum += stateAndProb[1] * (self.mdp.getReward(state, action, stateAndProb[0]) + self.discount * self.values[stateAndProb[0]])
              value = max(value, sum)
            if value > float('-inf'):
              nextValues[state] = value #update value in dictionary
      
        #after all values are up to date, next_state = curr_state
        for state in states:
          self.values[state] = nextValues[state]
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    statesAndProbs = self.mdp.getTransitionStatesAndProbs(state,action)
    q = 0
    for stateAndProb in statesAndProbs:
        q += stateAndProb[1] * (self.mdp.getReward(state, action, stateAndProb[0]) + self.discount * self.values[stateAndProb[0]])
    return q
    #util.raiseNotDefined()

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    actions = self.mdp.getPossibleActions(state)
    bestAction = None
    maxval = float('-inf')
    for action in actions:
      q = self.getQValue(state, action)
      if q > maxval:
        maxval = q
        bestAction = action
    return bestAction
    #util.raiseNotDefined()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
