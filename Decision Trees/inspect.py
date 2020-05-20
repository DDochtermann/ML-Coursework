# Written by Dan Dochtermann 1/31/2019

import sys
import numpy

if len(sys.argv) != 3:
	print("incorrect number of command line arguments, exiting")
	sys.exit()

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

#set column containing y
yCol = len(D[0])-1
numSamples = len(D)

#create dictionary of y labels and frequency
i=0
labels = {}
while i<len(D):
	if not D[i][yCol] in labels:
		labels[D[i][yCol]]=1
	else:
		labels[D[i][yCol]]+=1
	i+=1

#range over y's to get first node entropy
#find majority label
i=0
H = 0
majKey = ""
majCount = 0
for key in labels:
	H += -1*(labels[key]/numSamples)*numpy.log2(labels[key]/numSamples)
	if labels[key] > majCount:
		majKey = key
		majCount = labels[key]

#range over y's to calculate error
i=0
E=0
while i<len(D):
	if majKey != D[i][yCol]:
		E+=1	
	i+=1
E = E *(1/numSamples)

outFile = open(sys.argv[2], 'w')
outFile.write("entropy: " + str(H) + "\n")
outFile.write("error: " + str(E))
outFile.close()

#error = 1/N*sum{prediction != y}
#entropy = -sum{P(x)log2(P(x))