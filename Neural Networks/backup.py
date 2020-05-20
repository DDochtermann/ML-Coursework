# Written by Dan Dochtermann 3/20/19
# 10-601 HW#5
# AndrewID: DDochter

import sys
import numpy as np

class O:
	def __init__(self):
		self.x = []
		self.a = []
		self.b = []
		self.z = []
		self.yhat = []

def GetFileLines(num):
	inFile = open(sys.argv[num], 'r')
	fileLines = inFile.read().splitlines()
	inFile.close()
	return fileLines

def InitializeAB(hiddenunits, initflag):
	if initflag == 2:
		alpha = [[0 for i in range(0, 128)] for j in range(0, hiddenunits)]
		beta = [[0 for i in range(0, hiddenunits)] for j in range(0, 10)]

	if initflag == 1:
		alpha = [[np.random.uniform(low=-0.1, high=0.1) for i in range(0, 128)] for j in range(0, hiddenunits)]
		beta = [[np.random.uniform(low=-0.1, high=0.1) for i in range(0, hiddenunits)] for j in range(0, 10)]

	alpha = np.array(alpha)
	alpha = alpha.astype(float)
	beta = np.array(beta)
	beta = beta.astype(float)
	aBias = [[0] for i in range(0, hiddenunits)]
	alpha = np.hstack((aBias, alpha))
	bBias = [[0] for i in range(0, 10)]
	beta = np.hstack((bBias, beta))

	return (alpha, beta)

def ProcessFile(inFile):
	outData = []
	for i in range(0, len(inFile)):
		outData.append(inFile[i].split(","))
	return outData

def SGD(alpha, beta, trainData, testData, epochs, lr):
	Metrics = []
	for k in range(0, epochs):
		print("Processing epoch number", k +1 )
		for i in range(0, len(trainData)):
			yArr = [0 for q in range(0, 10)]
			yArr[int(trainData[i][0])] += 1
			yArr = np.array(yArr)
			y = yArr[:, np.newaxis]
			alpha, beta = SGDStep(alpha, beta, y, trainData[i][1:].astype(float), lr)
		MCtr, MCte = Evaluate(alpha, beta, trainData, testData)
		Metrics.append([k+1, MCtr, MCte])
	return (alpha, beta, Metrics)

def Evaluate(alpha, beta, trainData, testData):
	trainMCE = MCEntropy(alpha, beta, trainData)
	testMCE = MCEntropy(alpha, beta, testData)

	return (trainMCE, testMCE)

def MCEntropy(alpha, beta, inData):
	TotEntropy = 0
	N = len(inData)
	for i in range(0, N):
		yArr = [0 for q in range(0, 10)]
		yArr[int(inData[i][0])] += 1
		yArr = np.array(yArr)
		y = yArr[:, np.newaxis]
		x = inData[i][1:].astype(float)

		o = NNForward(x, y, alpha, beta)

		TotEntropy += np.log(o.yhat[int(inData[i][0])])
	MCE = -1*TotEntropy/N

	return MCE[0]

def CollectLabels(alpha, beta, inData):
	labels = []
	N = len(inData)
	for i in range(0, N):
		yArr = [0 for q in range(0, 10)]
		yArr[int(inData[i][0])] += 1
		yArr = np.array(yArr)
		y = yArr[:, np.newaxis]
		x = inData[i][1:].astype(float)

		o = NNForward(x, y, alpha, beta)

		labels.append(np.argmax(o.yhat))

	return labels

def SGDStep(alpha, beta, y, x, lr):
	
	o = NNForward(x, y, alpha, beta)
	gAlpha, gBeta = NNBackward(o.x, y, alpha, beta, o)

	alpha = alpha - (lr*gAlpha)

	bias = [[0] for i in range(0, 10)]
	beta = beta - (lr*gBeta)

	return (alpha, beta)

def NNForward(x, y, alpha, beta):
	o = O()
	x = np.insert(x, 0, 1)
	x = x[:, np.newaxis]
	o.x = x
	o.a = np.dot(alpha, o.x)
	o.z = Sigmoid(o.a)

	o.z = np.insert(o.z, 0, 1)
	o.z = o.z[:, np.newaxis]

	betaStar = np.delete(beta, 0, 1)
	o.b = np.dot(beta, o.z)
	o.yhat = SM(o.b)
	
	return o
	
def NNBackward(x, y, alpha, beta, o):

	betaStar = np.delete(beta, 0, 1)

	yDiff = np.subtract(o.yhat, y)
	gAlpha = np.dot(yDiff.T, betaStar)
	zz = np.multiply(o.z, (1-o.z))

	gAlpha = np.multiply(gAlpha.T, zz[1:])
	gAlpha = np.dot(gAlpha, x.T)

	gBeta = np.dot(yDiff, o.z.T)

	return gAlpha, gBeta

def SM(b):
	z = np.exp(b)
	return z/np.sum(z)

def Sigmoid(a):
	return 1/(1+np.exp(-a))

def WriteOutLabels(labels, fileName):
	outFile = open(fileName, 'w')
	for i in range(0, len(labels)):
		outFile.write("%i\n" % labels[i])
	outFile.close()

def CalcErrors(trainData, trainLabels, testData, testLabels):
	trainMarks = 0
	testMarks = 0
	trainN = len(trainData)
	testN = len(testData)

	for i in range(0, trainN):
		if int(trainData[i][0]) != int(trainLabels[i]):
			trainMarks += 1

	for i in range(0, testN):
		if int(testData[i][0]) != int(testLabels[i]):
			testMarks += 1

	trainErr = trainMarks/trainN
	testErr = testMarks/testN

	return (trainErr, testErr)

def WriteMetrics(Metrics, trainErr, testErr, fileName):
	outFile = open(fileName, 'w')

	for i in range(0, len(Metrics)):
		outFile.write("epoch=%i crossentropy(train): %.11f\n" % (Metrics[i][0], Metrics[i][1]))
		outFile.write("epoch=%i crossentropy(test): %.11f\n" % (Metrics[i][0], Metrics[i][2]))

	outFile.write("error(train): %.2f\n" % trainErr)
	outFile.write("error(test): %.2f" % testErr)
	outFile.close()

if __name__=="__main__":
	trainFile = GetFileLines(1)
	testFile = GetFileLines(2)
	trainData = np.array(ProcessFile(trainFile))
	trainData.astype(float)
	testData = np.array(ProcessFile(testFile))
	testData.astype(float)


	numepoch = int(sys.argv[6])
	hiddenunits = int(sys.argv[7])
	initflag = int(sys.argv[8])
	learningrate = float(sys.argv[9])
	
	alpha, beta = InitializeAB(hiddenunits, initflag)

	alpha, beta, Metrics = SGD(alpha, beta, trainData, testData, numepoch, learningrate)

	trainLabels = CollectLabels(alpha, beta, trainData)
	testLabels = CollectLabels(alpha, beta, testData)

	WriteOutLabels(trainLabels, sys.argv[3])
	WriteOutLabels(testLabels, sys.argv[4])
	trainErr, testErr = CalcErrors(trainData, trainLabels, testData, testLabels)

	WriteMetrics(Metrics, trainErr, testErr, sys.argv[5])


