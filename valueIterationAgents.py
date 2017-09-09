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

		for i in range(self.iterations):
			v = self.values.copy();
			for s in self.mdp.getStates():
				if self.mdp.isTerminal(s):
					self.values[s] = 0
				else:
					qvalues = []
					legalActions = self.mdp.getPossibleActions(s);
					if len(legalActions)<1:
						self.values[s] = 0;
					else:
						for a in legalActions:
							value = 0.0
							t = self.mdp.getTransitionStatesAndProbs(s, a)
							for i in range(len(t)):
								value += t[i][1] * (self.mdp.getReward(s, a, t[i][0]) + self.discount*v[t[i][0]])
							qvalues.append(value);
						maxValue = max(qvalues);
						self.values[s] = maxValue;

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

		if self.mdp.isTerminal(state): return 0;

		t = self.mdp.getTransitionStatesAndProbs(state, action);
		qvalue = 0;
		for i in range(len(t)):
			qvalue = qvalue + t[i][1] * (self.mdp.getReward(state, action, t[i][0]) + self.discount*self.values[t[i][0]])
		return qvalue;

		util.raiseNotDefined()

	def computeActionFromValues(self, state):
		"""
		  The policy is the best action in the given state
		  according to the values currently stored in self.values.

		  You may break ties any way you see fit.  Note that if
		  there are no legal actions, which is the case at the
		  terminal state, you should return None.
		"""
		"*** YOUR CODE HERE ***"
		if self.mdp.isTerminal(state): return None;

		legalMoves = self.mdp.getPossibleActions(state);
		values = []
		for a in legalMoves:
			values.append(self.computeQValueFromValues(state, a));
		return legalMoves[values.index(max(values))];

		util.raiseNotDefined()

	def getPolicy(self, state):
		return self.computeActionFromValues(state)

	def getAction(self, state):
		"Returns the policy at the state (no exploration)."
		return self.computeActionFromValues(state)

	def getQValue(self, state, action):
		return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
	"""
		* Please read learningAgents.py before reading this.*

		An AsynchronousValueIterationAgent takes a Markov decision process
		(see mdp.py) on initialization and runs cyclic value iteration
		for a given number of iterations using the supplied
		discount factor.
	"""
	def __init__(self, mdp, discount = 0.9, iterations = 1000):
		"""
		  Your cyclic value iteration agent should take an mdp on
		  construction, run the indicated number of iterations,
		  and then act according to the resulting policy. Each iteration
		  updates the value of only one state, which cycles through
		  the states list. If the chosen state is terminal, nothing
		  happens in that iteration.

		  Some useful mdp methods you will use:
			  mdp.getStates()
			  mdp.getPossibleActions(state)
			  mdp.getTransitionStatesAndProbs(state, action)
			  mdp.getReward(state)
			  mdp.isTerminal(state)
		"""
		ValueIterationAgent.__init__(self, mdp, discount, iterations)

	def runValueIteration(self):
		"*** YOUR CODE HERE ***"
		states = self.mdp.getStates();
		c = 0;
		for i in range(self.iterations):
			v = self.values.copy();
			if (c == len(states)): c = 0;
			s = states[c];
			if self.mdp.isTerminal(s):
				self.values[s] = 0
			else:
				qvalues = []
				legalActions = self.mdp.getPossibleActions(s);
				if len(legalActions)<1:
					self.values[s] = 0;
				else:
					for a in legalActions:
						value = 0.0
						t = self.mdp.getTransitionStatesAndProbs(s, a)
						for j in range(len(t)):
							value += t[j][1] * (self.mdp.getReward(s, a, t[j][0]) + self.discount*v[t[j][0]])
						qvalues.append(value);
					maxValue = max(qvalues);
					self.values[s] = maxValue;
			c = c +1;


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
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
		# if you see any similarity with other students code it is because I discussed my code during OH and 
		# since I was having problems do to the spacing in my code I simplified my code using other students ideas
		# however the original code that solved the problem correctly was writen by me and I DID NOT use any other sources
		# compute the predecessors of all states
		
		states = self.mdp.getStates();
		predecessor = util.Counter();

		for s in states:
			actions = self.mdp.getPossibleActions(s)
			for a in actions:
				for t in self.mdp.getTransitionStatesAndProbs(s,a):
					if t[1] != 0.0:
						if predecessor[t[0]] == 0:
							predecessor[t[0]] = set()
						else: 
							predecessor[t[0]].add(s)

		# initialiase pqueue to 0
		pq = util.PriorityQueue();

		for s in states:
			if self.mdp.isTerminal(s) == False:
				actions = self.mdp.getPossibleActions(s);
				qvalues = [self.computeQValueFromValues(s,a) for a in actions]

				maxQvalue = max(qvalues);
				diff = abs(self.values[s] - maxQvalue);
				pq.update(s, -diff);

		for i in range(self.iterations):
			if pq.isEmpty(): return
			s = pq.pop();

			#update the value of the state in self.values
			actions = self.mdp.getPossibleActions(s);
			qvalues = [self.computeQValueFromValues(s,a) for a in actions]
			self.values[s] = max(qvalues);

			for p in predecessor[s]:
				moves = self.mdp.getPossibleActions(p);
				pvalues = [self.computeQValueFromValues(p,a) for a in moves]
				diff = abs(max(pvalues) - self.values[p])
				if diff> self.theta:
					pq.update(p, -diff)

