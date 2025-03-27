# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise

def question3a():
    # Prefer the close exit (+1), risking the cliff (-10)
    # A relatively high discount keeps the exit rewarding enough,
    # noise = 0.0 allows a risky path along the cliff,
    # and a more negative living reward encourages finishing quickly.
    answerDiscount = 0.1
    answerNoise = 0.0
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    # Prefer the close exit (+1), but avoiding the cliff (-10)
    # We keep a high discount but introduce some noise so the agent
    # becomes more cautious, and a negative living reward encourages
    # a reasonably quick exit.
    answerDiscount = 0.1
    answerNoise = 0.1
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    # Prefer the distant exit (+10), risking the cliff (-10)
    # A high discount (so the distant reward is valuable), moderate noise,
    # and a moderately negative living reward push the agent to go for +10.
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    # Prefer the distant exit (+10), avoiding the cliff (-10)
    # High discount for the distant exit, enough noise to avoid the direct risky path,
    # and a living reward of 0 ensures the agent doesn't prefer finishing too quickly.
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    # Avoid both exits and the cliff (so an episode never terminates)
    # A positive living reward makes it worthwhile to keep wandering,
    # rather than ever exiting.
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 1
    return answerDiscount, answerNoise, answerLivingReward

def question8():
    answerEpsilon = None
    answerLearningRate = None
    if answerEpsilon or answerLearningRate:
        return answerEpsilon, answerLearningRate
    return "NOT POSSIBLE"
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))