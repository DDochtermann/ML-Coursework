# Written by Dan Dochtermann 1/31/2019

import sys
import numpy

# takes dataset and column index to calculate entropy on
def Entropy(D, col):
	H = 0
	numSamples = 0

	labels = Probability(D, col, "none", "none")
	for key in labels:
		numSamples += labels[key]
	for key in labels:
		H += -1*(labels[key]/numSamples)*numpy.log2(labels[key]/numSamples)

	return H
###

# takes dataset and column indeces to calc conditional entropy of a|b
def CondEntropy(D, aCol, bCol):
	H = 0
	aSamples = 0
	bSamples = 0

	aLabels = Probability(D, aCol, "none", "none")
	bLabels = Probability(D, bCol, "none", "none")
	
	for keyB in bLabels: 
		bSamples += bLabels[keyB]

	for keyA in aLabels: 
		aSamples += aLabels[keyA]

	for keyB in bLabels:
		temp = 0
		for keyA in aLabels:
			if SingleProbability(D, aCol, keyA, bCol, keyB) != 0:
				temp += SingleProbability(D, aCol, keyA, bCol, keyB)*numpy.log2(SingleProbability(D, aCol, keyA, bCol, keyB))
		H += -1*(bLabels[keyB]/bSamples)*temp

	return H
###

def SingleProbability(D, col, state, condCol, condState):
	labels = Probability(D, col, condCol, condState)
	numSamples = 0
	for key in labels:
		numSamples += labels[key]
	if not state in labels :
		return 0
	else:
		return (labels[state]/numSamples)
###

# finds the probability dictionary of an x in D (optionally, given a condition)
def Probability(D, col, condCol, condState):
	P=0
	numSamples = len(D)
	i = 0
	labels = {}

	if condCol == "none":
		while i<len(D):
			if not D[i][col] in labels:
				labels[D[i][col]]=1
			else:
				labels[D[i][col]]+=1
			i+=1
	else:
		while i<len(D):
			if D[i][condCol] == condState:
				if not D[i][col] in labels:
					labels[D[i][col]]=1
				else:
					labels[D[i][col]]+=1
			i+=1
	return labels
###

def MutInf(D, aCol, bCol):
	return Entropy(D, aCol) - CondEntropy(D, aCol, bCol)
###


inFile = open(sys.argv[1], 'r')
fileLines = inFile.readlines()
inFile.close()

#create matrix D for file data
D = []
i=1
while i < len(fileLines):
	line = fileLines[i].split(",")
	D.append(line)
	i+=1

print(MutInf(D, 3, 0))