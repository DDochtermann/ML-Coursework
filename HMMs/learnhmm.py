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

def GetPriors(N, I2T, trainFile):
	D = np.ones(N)
	M = len(trainFile)
	for i in range(0, M):
		a = trainFile[i].split(" ")
		b = a[0].split("_")
		c = b[1]
		D[I2T[c]-1] += 1
	D = np.divide(D, (N + M))
	
	return D

def WritePriors(P, fileName):
	outFile = open(fileName, 'w')

	for i in range(0, len(P)):
		outFile.write("%.18e\n" % P[i])

def WriteTrans(P, fileName):
	outFile = open(fileName, 'w')

	for i in range(0, len(P)):
		for j in range(0, len(P[i])):
			outFile.write("%.18e" % P[i][j])
			if j < len(P[i])-1:
				outFile.write(" ")
		outFile.write("\n")

def GetTransition(N, I2T, trainFile):
	D = np.ones((N, N))
	M = len(trainFile)
	for i in range(0, M):
		a = trainFile[i].split(" ")
		for j in range(0, len(a)-1):
			b = a[j].split("_")
			c = b[1]
			d = a[j+1].split("_")
			e = d[1]
			D[I2T[c]-1][I2T[e]-1] += 1

	div = np.zeros((N,1))
	for i in range(0, N):
		for j in range(0, N):
			div[i] += D[i][j]

	D = D/div
	
	return D

def GetEmission(tN, I2T, wN, I2W, trainFile):
	D = np.ones((tN, wN))
	
	M = len(trainFile)
	for i in range(0, M):
		a = trainFile[i].split(" ")
		for j in range(0, len(a)):
			b = a[j].split("_")
			w = b[0]
			t = b[1]
			D[I2T[t]-1][I2W[w]-1] += 1

	div = np.zeros((tN, 1))
	for i in range(0, len(D)):
		for j in range(0, len(D[i])):
			div[i] += D[i][j]

	D = D/div
	return D
		
if __name__ == "__main__":

	trainFile = GetFileLines(1)
	indexToWordFile = GetFileLines(2)
	indexToTagFile = GetFileLines(3)

	I2W, wordCount = ProcessIndexFile(indexToWordFile)
	I2T, tagCount = ProcessIndexFile(indexToTagFile)

	P = GetPriors(tagCount, I2T, trainFile)
	WritePriors(P, sys.argv[4])
	A = GetTransition(tagCount, I2T, trainFile)
	WriteTrans(A, sys.argv[6])
	B = GetEmission(tagCount, I2T, wordCount, I2W, trainFile)
	WriteTrans(B, sys.argv[5])