# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import mdp, random, util, math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    self.qvalues = util.Counter() # A Counter is a dict with default 0 #holds values of current state

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
    q = self.qvalues[(state, action)]
    # how do I get 0.0 if have never seen?
    return q
    #util.raiseNotDefined()


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    actions = self.getLegalActions(state)
    max_action = 0.0
    if len(actions) > 0:
      for action in actions:
        actionValue = self.getQValue(state,action)
        if actionValue > max_action:
          max_action = actionValue
    return max_action
    #util.raiseNotDefined()

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    actions = self.getLegalActions(state)
    bestAction = None
    maxval = float('-inf')
    for action in actions:
      q = self.getQValue(state, action)
      if (q > maxval):
        maxval = q
        bestAction = action
    return bestAction
    #util.raiseNotDefined()

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = self.getPolicy(state)
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    p = self.epsilon
    whichAction = util.flipCoin(p) #simulate binary varible with probability p of success
    #returns True with probability p: want to do random action
    if (whichAction == True):
      action = random.choice(legalActions)
    #if returns False: want to taek best policy action
    # don't need to change action? or do else statement & define it as best one
    # but I think it might already be best one
    
    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    #value = reward + gamme * #get qvalue
    
    oldQ = self.getQValue(state, action)
    maxQprime = self.getValue(nextState)

    #oldFeatureVector = self.featExtractor.getFeautures(state, action)

    gamma = self.discount
    #difference = (reward + gamma * maxQprime) - oldQ
    alpha = self.alpha

    #newQ = oldQ + alpha*difference
    newQ = ((1-alpha)*oldQ) + (alpha*(reward + gamma*maxQprime))
    self.qvalues[(state,action)] = newQ

    #for feature in oldFeatureVector:
    #  self.weights[feature] += a * difference * oldFeatureVector[feacture]
    

    #util.raiseNotDefined()

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weights = util.Counter() #dict with default 0, holds values of weights

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"

    w = self.weights
    featureVector = self.featExtractor.getFeatures(state, action) #dict (feature vector)
    
    return w * featureVector
    #util.raiseNotDefined()

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"

    oldQ = self.getQValue(state, action) #get old Q value
    newQ = self.getValue(nextState) #have to get Value, not QValue, because don't have action?
    # also because getValue gets max Qvalue

    oldFeatureVector = self.featExtractor.getFeatures(state, action)
    
    gamma = self.discount
    difference = (reward + gamma*newQ) - oldQ
    alpha = self.alpha

    #update weight of each feature
    for feature in oldFeatureVector:
      #print 'self.weights[feature]: ', self.weights[feature]
      self.weights[feature] += alpha * difference * oldFeatureVector[feature]
    #util.raiseNotDefined()

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
