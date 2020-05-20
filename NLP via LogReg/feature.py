# Written by Dan Dochtermann 2/27/19
# 10-601 HW#4
# AndrewID: DDochter

import sys
import numpy as np
import collections

def GetFileLines(num):
	inFile = open(sys.argv[num], 'r')
	fileLines = inFile.read().splitlines()
	inFile.close()
	return fileLines

def PrintLineToFile(y, outDict, outFile):
	outFile.write("%s\t" % y)
	for i in outDict:
		outFile.write("%s:1\t" % i)
	outFile.write("\n")
	
def ProcessDict(dictFile):
	wordDict = []
	for i in range(0, len(dictFile)):
		a = dictFile[i].split(" ")
		wordDict.append(a[0])
	return wordDict

def FormatFile(wordDict, trainFile, fFlag, outNum):
	outFile = open(sys.argv[outNum], 'w')
	for i in range(0, len(trainFile)):
		outDict = collections.OrderedDict()
		a = trainFile[i].split("	")
		b = a[1].split(" ")
		for j in range(0, len(b)):
			if b[j] in wordDict:
				if wordDict.index(b[j]) not in outDict:	
					outDict[wordDict.index(b[j])] = 1
				else:
					outDict[wordDict.index(b[j])] += 1

		if fFlag == 2:
			for k in outDict.copy():
				if outDict[k] >= 4:
					del outDict[k]

		PrintLineToFile(a[0], outDict, outFile)
	outFile.close

if __name__=="__main__":

	trainFile = GetFileLines(1)
	validFile = GetFileLines(2)
	testFile = GetFileLines(3)
	dictFile = GetFileLines(4)
	fFlag = int(sys.argv[8])

	#Process dictFile input so that each word index matches it's value in the file
	wordDict = ProcessDict(dictFile)

	#Process input files and output to formatted versions
	FormatFile(wordDict, trainFile, fFlag, 5)
	print("train formatting complete")
	FormatFile(wordDict, validFile, fFlag, 6)
	print("valid formatting complete")
	FormatFile(wordDict, testFile, fFlag, 7)
	print("test formatting complete")
