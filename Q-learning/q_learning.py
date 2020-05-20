from environment import MountainCar
import sys
import numpy as np

def main(args):
	mode = args[1]
	weightFile = args[2]
	returnsFile = args[3]
	episodes = int(args[4])
	max_iter = int(args[5])
	epsilon = float(args[6])
	gamma = float(args[7])
	alpha = float(args[8])

	M = MountainCar(mode)
	M.reset()
	W = np.zeros((M.state_space, 3))
	b = 0
	rewards = [0 for i in range(0, episodes)]

	for i in range(0, episodes):
		sDict = M.reset()
		s = [0 for k in range(0, M.state_space)]
		for j in sDict:
			s[j] = sDict[j]
		iters = 0
		stepResult = [0,0,False]
		while stepResult[2] != True and iters < max_iter:
			s0 = s
			a = Action(s0, W, epsilon, b)
			q = Qval(s0, a, W, b)
			stepResult = M.step(a)
	
			r = stepResult[1]
			sDict = stepResult[0]
			s = [0 for k in range(0, M.state_space)]
			for j in sDict:
				s[j] = sDict[j]
			rewards[i]+=r
			q2s = [Qval(s,0,W,b), Qval(s,1,W,b), Qval(s,2,W,b)]
			a2 = q2s.index(max(q2s))
		
			gW = np.zeros((M.state_space, 3))
			gW[:,a] = s0
			
			x = alpha*(q - (r + gamma * Qval(s, a2, W, b)))
			W = W - x*gW
			b = b - x
			iters+=1

	WriteWeights(b, W, weightFile)
	WriteRewards(rewards, returnsFile)

def WriteWeights(b, W, WFile):
	outFile = open(WFile, 'w')
	outFile.write("%.19f\n" % b)
	for i in range(0, len(W)):
		for j in range(0, len(W[i])):
			outFile.write("%.19f\n" % W[i][j])
	outFile.close()

def WriteRewards(r, RFile):
	outFile = open(RFile, 'w')
	for i in range(0, len(r)):
		outFile.write("%.1f\n" % r[i])
	outFile.close()

def Action(s, w, epsilon, b):
	v = np.random.ranf(1)
	if v < 1-epsilon:
		Qs = [Qval(s,0,w,b), Qval(s,1,w,b), Qval(s,2,w,b)]
		a = Qs.index(max(Qs))
	else:
		a = np.random.choice([0,1,2])
	return a
		
def Qval(s, a, w, b):
	return np.dot(np.array(s).T,w[:,a])+b

if __name__ == "__main__":
	main(sys.argv)