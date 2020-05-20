# Written by Dan Dochtermann 4/9/19
# 10-601 HW#7
# AndrewID: DDochter

import sys
import numpy as np

def GetFileLines(num):
	inFile = open(sys.argv[num], 'r')
	fileLines = inFile.read().splitlines()
	inFile.close()
	return fileLines

def ProcessIndexFile(file):
	D = {}
	wordCount = len(file)
	for i in range(0, wordCount):
		D[file[i]] = i+1
	return (D, wordCount)

def ReverseIndexFile(file):
	D = {}
	wordCount = len(file)
	for i in range(0, wordCount):
		D[i+1] = file[i]
	return (D, wordCount)

def ProcessPrior(file):
	P = np.zeros((len(file), 1))
	for i in range(0, len(file)):
		P[i] = float(file[i])
	return P

def ProcessET(file):
	A = []
	for i in range(0, len(file)):
		a = file[i].split(" ")
		b = []
		for j in range(0, len(a)):
			b.append(float(a[j]))
		A.append(b)
	A = np.array(A)
	return A

def ForwardStep(P, A, B, N, M, words, tags, W2I, T2I):
	alpha = np.zeros((N, M))
	for j in range(0, len(alpha)):
		alpha[j][0] = P[j]*B[j][W2I[words[0]]-1]
	for t in range(1, M):
		for j in range(0, len(alpha)):
			alpha[j][t] = B[j][W2I[words[t]]-1]*FowSum(alpha, t-1, A, j, N)

	return alpha

def FowSum(alpha, t, A, j, N):
	tot = 0
	for k in range(0, N):
		tot += alpha[k][t]*A[k][j]
	return tot

def BackwardStep(P, A, B, N, M, alpha, words, tags, W2I, T2I):
	beta = np.zeros((N, M))
	for j in range(0, len(alpha)):
		beta[j][M-1] = 1
	t = M - 2
	while t >= 0:
		for j in range(0, len(alpha)):
			beta[j][t] = BackSum(B, beta, A, t+1, j, words, W2I, N)
		t -= 1

	return beta

def BackSum(B, beta, A, t, j, words, W2I, N):
	tot = 0
	for k in range(0, N):
		tot += B[k][W2I[words[t]]-1]*beta[k][t]*A[j][k]
	return tot

def Predict(alpha, beta, I2T):
	P = np.multiply(alpha, beta)
	print("matrix product")
	print(P)

	PP = np.argmax(P, axis = 0)
	print("argmax")
	print(PP)

	yhat = ["" for i in range(0, len(PP))]
	for i in range(0, len(yhat)):
		yhat[i] = I2T[PP[i]+1]

	print("yhat")
	print(yhat)

	return yhat

def ProcessTestFile(file):
	testWords = []
	testTags = []
	for i in range(0, len(file)):
		wordLine = []
		tagLine = []
		a = file[i].split(" ")
		for j in range(0, len(a)):
			q = a[j].split("_")
			wordLine.append(q[0])
			tagLine.append(q[1])
		testWords.append(wordLine)
		testTags.append(tagLine)
	return (testWords, testTags)

def GetLineString(tags, words):
	line = ""
	i = -1
	for i in range(0, len(words)-1):
		line += (words[i] + "_" + tags[i] + " ")
	i += 1
	line += (words[i] + "_" + tags[i])
	return line

def WritePredictedFile(lines, fileName):
	outFile = open(fileName, 'w')
	i=-1
	for i in range(0, len(lines)-1):
		outFile.write("%s\n" % lines[i])
	i+=1
	outFile.write("%s" % lines[i])
	outFile.close()

def LogLikelyhood(alpha, T):
	tot = 0
	for j in range(0, T):
		tot += alpha[j][len(alpha[0])-1]

	if tot == 0:
		return -120
	else:
		return np.log(tot)

def GetAccuracy(predicted, actual):
	count = 0
	match = 0
	for i in range(0, len(predicted)):
		for j in range(0, len(predicted[i])):
			count += 1
			if predicted[i][j] == actual[i][j]:
				match += 1

	return match/count

def WriteMetrics(avgLL, acc, fileName):
	outFile = open(fileName, 'w')

	outFile.write("Average Log-Likelihood: %.8f\n" % avgLL)
	outFile.write("Accuracy: %.12f\n" % acc)
	outFile.close()

if __name__ == "__main__":

	testFile = GetFileLines(1)
	indexToWordFile = GetFileLines(2)
	indexToTagFile = GetFileLines(3)

	W2I, wordCount = ProcessIndexFile(indexToWordFile)
	T2I, tagCount = ProcessIndexFile(indexToTagFile)
	I2T, _ = ReverseIndexFile(indexToTagFile)

	priorFile = GetFileLines(4)
	emitFile = GetFileLines(5)
	transFile = GetFileLines(6)

	P = ProcessPrior(priorFile)
	A = ProcessET(transFile)
	B = ProcessET(emitFile)

	testWords, testTags = ProcessTestFile(testFile)

	predictedAll = []
	fullTags = []
	llTot = 0
	
	for i in range(0, len(testFile)):

		if i%250 == 0:
			print(i, "of", len(testFile)-1)

		alpha = ForwardStep(P, A, B, tagCount, len(testWords[i]), testWords[i], testTags[i], W2I, T2I)
		print("alpha")
		print(alpha)

		beta = BackwardStep(P, A, B, tagCount, len(testWords[i]), alpha, testWords[i], testTags[i], W2I, T2I)
		print("beta")
		print(beta)

		predictedTags = Predict(alpha, beta, I2T)

		fullTags.append(predictedTags)

		predictedAll.append(GetLineString(predictedTags, testWords[i]))

		llTot += LogLikelyhood(alpha, tagCount)

	WritePredictedFile(predictedAll, sys.argv[7])

	AvgLL = llTot/len(testFile)

	Accuracy = GetAccuracy(fullTags, testTags)

	WriteMetrics(AvgLL, Accuracy, sys.argv[8])
