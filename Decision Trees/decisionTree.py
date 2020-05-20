# Written by Dan Dochtermann 1/31/2019

import sys
import numpy


class Node:
	def __init__(self,name):
		self.children = []
		self.path = ""
		self.name = name
		self.depth = 0
		self.data = []
		self.majority = ""

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

def PrintoutNodeContents(D, col):
	labels = Probability(D, col, "none", "none")
	i=0
	datastring = "["
	for key in labels:
		datastring += (str(labels[key]) + "x " + key)
		if i != len(labels)-1:
			datastring += " / "
		i+=1
	return datastring + "]"

###

def MutInf(D, aCol, bCol):
	return Entropy(D, aCol) - CondEntropy(D, aCol, bCol)
###

def TrimDataset(D, colInx, colState):
	i=0
	Trimmed = []
	while i<len(D):
		if D[i][colInx] == colState:
			Trimmed.append(D[i])
		i+=1
	return Trimmed
###

def SplitOnMutInf(D, Conditions, depth, maxDepth, parentNode):
	i=0
	currentI = 0
	colInx = -1

	while i<len(D[0])-1:
		if MutInf(D,i,len(D[0])-1) > currentI:
			currentI = MutInf(D,i,len(D[0])-1)
			colInx = i
		i+=1

	j=0
	states = []
	while j<len(D):
		if not D[j][colInx] in states:
			states.append(D[j][colInx])
		j+=1
	
	if currentI != 0 and depth < maxDepth:
		depth += 1
		parentNode.name = Conditions[colInx]
		for key in states:
			nextChild = Node("leaf")
			nextChild.path = key
			nextChild.depth = depth
			T = TrimDataset(D,colInx,key)
			nextChild.data = T
			nextChild.majority = MajoirtyClass(T, len(D[0])-1)
			parentNode.children.append(nextChild)
			SplitOnMutInf(T, Conditions, depth, maxDepth, nextChild)

	else:
		k = 0
		leaflabels = {}
		while k<len(D):
			if not D[k][len(D[0])-1] in leaflabels:
				 leaflabels[D[k][len(D[0])-1]]=1
			else:
				 leaflabels[D[k][len(D[0])-1]]+=1
			k+=1

		majority = ""
		majCount = 0
		for key in leaflabels:
			if leaflabels[key] > majCount:
				majCount = leaflabels[key]
				majority = key
		parentNode.majority = MajoirtyClass(D, len(D[0])-1)
###

def PrintNodes(N, L):

	print(PrintoutNodeContents(N.data, L), N.majority)
	i=0
	while i<len(N.children):
		j=0
		while j<N.depth+1:
			print("|", end='')
			j+=1
		print(N.name, "=", N.children[i].path, ":", end='')
		PrintNodes(N.children[i], L)
		i+=1
###

def MajoirtyClass(N, L):
	leaflabels = Probability(N, L, "none", "none")
	majority = ""
	majCount = 0
	for key in leaflabels:
		if leaflabels[key] > majCount:
			majCount = leaflabels[key]
			majority = key
	return majority
###

def PredictFromTree(testD, root, Conditions, outFile):
	i = 0
	while i<len(testD):
		GetPrediction(testD[i],root, Conditions, outFile)
		i+=1

###

def GetPrediction(dataline, root, Conditions, outFile):
	j=0
	if root.name == "leaf" or len(root.children) == 0:
		outFile.write(root.majority + "\n")
	else: 
		while j<len(Conditions):
			if root.name == Conditions[j]:
				x = dataline[j]
				for k in range(0, len(root.children)):
					if root.children[k].path == x:
						GetPrediction(dataline,root.children[k],Conditions, outFile)
			j+=1
###

def PrintMetrics(trainIn, trainOutFile, testIn, testOutFile, outFile, L):
	trainSum = 0
	testSum = 0
	trainOut = open(trainOutFile, 'r').read().splitlines()
	testOut = open(testOutFile, 'r').read().splitlines()

	for i in range(0, len(trainOut)):
		if trainOut[i] != trainIn[i][L]:
			trainSum += 1

	for i in range(0, len(testOut)):
		if testOut[i] != testIn[i][L]:
			testSum += 1


	outFile.write("error(train): " + str(trainSum/len(trainOut)))
	outFile.write("\n")
	outFile.write("error(test): " + str(testSum/len(testOut)))

###


inFile = open(sys.argv[1], 'r')
fileLines = inFile.read().splitlines()
inFile.close()

#create matrix D for file data
D = []
Conditions = fileLines[0].split(",")
i=1
while i < len(fileLines):
	line = fileLines[i].split(",")
	D.append(line)
	i+=1

inFile2 = open(sys.argv[2], 'r')
fileLines2 = inFile2.read().splitlines()
inFile2.close()
testD = []
i=1
while i < len(fileLines2):
	line = fileLines2[i].split(",")
	testD.append(line)
	i+=1

maxDepth = int(sys.argv[3])

root = Node("root")
root.data = D
SplitOnMutInf(D, Conditions, 0, maxDepth, root)
L = len(D[0])-1
PrintNodes(root, L)

outFile1 = open(sys.argv[4], 'w')
PredictFromTree(D, root, Conditions, outFile1)
outFile1.close()

outFile2 = open(sys.argv[5], 'w')
PredictFromTree(testD, root, Conditions, outFile2)
outFile2.close()

outFile3 = open(sys.argv[6], 'w')
PrintMetrics(D, sys.argv[4], testD, sys.argv[5], outFile3, L)
outFile3.close()

