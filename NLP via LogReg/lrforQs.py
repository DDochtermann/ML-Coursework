# Written by Dan Dochtermann 2/27/19
# 10-601 HW#4
# AndrewID: DDochter

import sys
import numpy as np

class dp:
	def __init__(self):
		self.y = 0
		self.x = []

def GetFileLines(num):
	inFile = open(sys.argv[num], 'r')
	fileLines = inFile.read().splitlines()
	inFile.close()
	return fileLines

def ProcessDict(dictFile):
	wordDict = []
	for i in range(0, len(dictFile)):
		a = dictFile[i].split(" ")
		wordDict.append(a[0])
	return wordDict

def ProcessInput(inFile, w):
	z = []

	for i in range(0, len(inFile)):
		nextUp = dp()
		nextUp.x = []
		a = inFile[i].split("	")
		nextUp.y = float(a[0])
		for j in range(1, len(a)):
			b = a[j].split(":")
			if b[0] != "":
				nextUp.x.append(int(b[0])+1)
		nextUp.x.append(0)
		z.append(nextUp)
	return z

def SGD(theta, testData, lr, epochs):
	for k in range(0, epochs):
		for i in range(0, len(testData)):
			theta = SGDStep(theta, testData[i].y, testData[i].x, lr)
	return theta

def SGDStep(theta, y, x, lr):
	expDP = np.exp(BetterDotProduct(theta, x))
	for j in range(0, len(x)):
			theta[x[j]] = theta[x[j]] + (lr * ( y - (expDP / (1+expDP)))	)
	return theta

def BetterDotProduct(t, x):
	s = 0
	for i in range(0, len(x)):
		s += t[x[i]]
	return s

def LogLikelyhood(theta, x):
	return np.exp(BetterDotProduct(theta, x))/(1+np.exp(BetterDotProduct(theta,x)))

def LabelPrediction(theta, inData):
	labels = []
	for i in range(0, len(inData)):
		if LogLikelyhood(theta, inData[i].x) < 0.5:
			labels.append(0)
		else:
			labels.append(1)
	return labels

def WriteLabelstoFile(labels, argNum):
	outFile = open(sys.argv[argNum], 'w')
	for i in range(0, len(labels)):
		outFile.write("%i\n" % labels[i])
	outFile.close()

def ErrorRate(labels, data):
	numPoints = len(labels)
	wrongs = 0.0
	for i in range(0, numPoints):
		if labels[i] != data[i].y:
			wrongs += 1
	return wrongs/float(numPoints)

def WriteMetrics(trainErr, testErr, argNum):
	outFile = open(sys.argv[argNum], 'w')
	outFile.write("error(train): %.6f\n" % trainErr)
	outFile.write("error(test): %.6f\n" % testErr)
	outFile.close()

def Q121(wordDict, trainData, validData):
	epochs = 200
	lr = 0.1
	theta = [0.0 for i in range(0, len(wordDict)+1)]
	theta = np.array(theta)
	theta = SGDWithReadouts(theta, trainData, lr, epochs, validData)

def AvgNegLL(theta, lr, inData):
	dataPts = len(inData)
	LLsum = 0
	for i in range(0, dataPts):
		BDP = BetterDotProduct(theta, inData[i].x)
		LLsum += -1 * inData[i].y * BDP + np.log(1 + np.exp(BDP)) 
	return LLsum/float(dataPts)


def SGDWithReadouts(theta, trainData, lr, epochs, validData):
	print("epoch # \t train \t valid")
	for k in range(0, epochs):
		for i in range(0, len(testData)):
			theta = SGDStep(theta, testData[i].y, testData[i].x, lr)
		avgTrainNegLL = AvgNegLL(theta, lr, trainData) 
		avgValidNegLL = AvgNegLL(theta, lr, validData)
		print(k+1, "\t", avgTrainNegLL, "\t", avgValidNegLL)

	return theta

if __name__=="__main__":

	trainFile = GetFileLines(1)
	validFile = GetFileLines(2)
	testFile = GetFileLines(3)

	dictFile = GetFileLines(4)
	wordDict = ProcessDict(dictFile)

	trainData = ProcessInput(trainFile, len(wordDict))
	validData = ProcessInput(validFile, len(wordDict))
	testData = ProcessInput(testFile, len(wordDict))

	Q121(wordDict, trainData, validData)
	"""
	epochs = int(sys.argv[8])
	lr = 0.1

	theta = [0.0 for i in range(0, len(wordDict)+1)]
	theta = np.array(theta)
	theta = SGD(theta, trainData, lr, epochs)

	trainLabels = LabelPrediction(theta, trainData)
	testLabels = LabelPrediction(theta, testData)

	WriteLabelstoFile(trainLabels, 5)
	WriteLabelstoFile(testLabels, 6)

	trainErr = ErrorRate(trainLabels, trainData)
	testErr = ErrorRate(testLabels, testData)

	WriteMetrics(trainErr, testErr, 7)
	"""



