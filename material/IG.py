# -*- coding: utf-8 -*-

# calculating the Entropy and Information Gain for: Learning with Trees
# by: Aziz Alto

# see Information Gain:
    # http://www.autonlab.org/tutorials/infogain.html


from __future__ import division
from math import log


def entropy(pi):
    '''
    return the Entropy of a probability distribution:
    entropy(p) = − SUM (Pi * log(Pi) )

    defintion:
            entropy is a metric to measure the uncertainty of a probability distribution.

    entropy ranges between 0 to 1

    Low entropy means the distribution varies (peaks and valleys).
    High entropy means the distribution is uniform.

    See:
            http://www.cs.csi.cuny.edu/~imberman/ai/Entropy%20and%20Information%20Gain.htm
    '''

    total = 0
    for p in pi:
        p = p / sum(pi)
        if p != 0:
            total += p * log(p, 2)
        else:
            total += 0
    total *= -1
    return total


def gain(d, a):
    '''
    return the information gain:
    gain(D, A) = entropy(D)−􏰋 SUM ( |Di| / |D| * entropy(Di) )
    '''

    total = 0
    for v in a:
        total += sum(v) / sum(d) * entropy(v)

    gain = entropy(d) - total
    return gain


# TEST

###__ example 1 (AIMA book, fig18.3)

# set of example of the dataset
willWait = [6, 6] # Yes, No

# attribute, number of members (feature)
patron = [ [4,0], [2,4], [0,2] ] # Some, Full, None

print(gain(willWait, patron))


###__ example 2 (playTennis homework)

# set of example of the dataset
playTennis = [9, 5] # Yes, No

# attribute, number of members (feature)
outlook = [
    [4, 0],  # overcase
    [2, 3],  # sunny
    [3, 2]   # rain
]
temperature = [
    [2, 2],  # hot
    [3, 1],  # cool
    [4, 2]   # mild
]
humidity = [
    [3, 4],  # high
    [6, 1]   # normal
]
wind = [
    [6, 2],  # weak
    [3, 3]   # strong
]

print(gain(playTennis, outlook))
print(gain(playTennis, temperature))
print(gain(playTennis, humidity))
print(gain(playTennis, wind))