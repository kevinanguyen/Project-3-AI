# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # If no iterations are specified, do nothing.
        if self.iterations == 0:
            return
        
        for _ in range(self.iterations):
            # Temporary Counter to store updated values for this iteration
            nextValues = util.Counter()

            # Loop over all states in the MDP
            for state in self.mdp.getStates():
                # If it's a terminal state, its value is 0
                if self.mdp.isTerminal(state):
                    nextValues[state] = 0
                else:
                    # Compute the maximum Q-value over all possible actions
                    maxQValue = float('-inf')
                    for action in self.mdp.getPossibleActions(state):
                        qValue = self.computeQValueFromValues(state, action)
                        if qValue > maxQValue:
                            maxQValue = qValue
                    nextValues[state] = maxQValue
            
            # Update self.values with the newly computed values for this iteration
            self.values = nextValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        # Sum over all possible next states
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.values[nextState])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get possible actions for this state
        possibleActions = self.mdp.getPossibleActions(state)
        if len(possibleActions) == 0:
            return None  # e.g., terminal state

        bestAction = None
        maxQValue = float('-inf')

        # Choose the action with the highest Q-value
        for action in possibleActions:
            qValue = self.computeQValueFromValues(state, action)
            if qValue > maxQValue:
                maxQValue = qValue
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
            # 1) Compute predecessors of all states
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)

        # 2) Initialize an empty priority queue
        pq = util.PriorityQueue()

        # 3) For each non-terminal state, calculate diff and push with priority -diff
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                currentValue = self.values[state]
                # Compute the best Q-value among all actions from this state
                bestQ = max(
                    self.computeQValueFromValues(state, action)
                    for action in self.mdp.getPossibleActions(state)
                )
                diff = abs(currentValue - bestQ)
                pq.push(state, -diff)  # negative because we want max diff to have highest priority

        # 4) Main iteration loop
        for i in range(self.iterations):
            if pq.isEmpty():
                break

            # Pop highest-priority state
            state = pq.pop()

            # If it's not terminal, update its value
            if not self.mdp.isTerminal(state):
                bestQ = max(
                    self.computeQValueFromValues(state, action)
                    for action in self.mdp.getPossibleActions(state)
                )
                self.values[state] = bestQ

            # For each predecessor p of this state...
            for p in predecessors[state]:
                if not self.mdp.isTerminal(p):
                    currentValue = self.values[p]
                    # Recompute the best possible Q-value for p
                    bestQ_p = max(
                        self.computeQValueFromValues(p, action)
                        for action in self.mdp.getPossibleActions(p)
                    )
                    diff = abs(currentValue - bestQ_p)
                    # If our change is large enough, push or update p in the priority queue
                    if diff > self.theta:
                        pq.update(p, -diff)  # update will push if p not there, or lower priority if -diff is better