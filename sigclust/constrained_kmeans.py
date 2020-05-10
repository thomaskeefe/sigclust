#
# Author: Stanislaw Adaszewski, 2015
#

import networkx as nx
import numpy as np
import time
from numpy import array, tile, concatenate
from numpy.linalg import norm

class ConstrainedKMeans(object):
	def __init__(self):
		pass

	def fit(self, data, demand):
		C, M, f = constrained_kmeans(data, demand)
		self.centroids = C
		self.labels = M
		self.flow_cost = f

def constrained_kmeans(data, demand, maxiter=None, fixedprec=1e9):
	data = array(data)

	min_ = np.min(data, axis=0)
	max_ = np.max(data, axis=0)

	n, d = data.shape
	K = len(demand)

	# Initialze centroids
	C = min_ + np.random.random((K, d)) * (max_ - min_)
	M = array([-1] * n, dtype=np.int)

	itercnt = 0
	while True:
		itercnt += 1
		g = nx.DiGraph()

		# Add nodes for each point 0...n-1
		g.add_nodes_from(range(0, n), demand=-1)

		# Add nodes for centroids
		for k in range(K):
			g.add_node(n + k, demand=demand[k])

		# Calculating cost...
		cost = array([norm(tile(data.T, K).T - tile(C, n).reshape(K * n, d), axis=1)])
		# Preparing data_to_C_edges...
		data_to_C_edges = concatenate(
		                      (tile([range(n)], K).T,
							   tile(array([range(n, n + K)]).T, n).reshape(K * n, 1),
							   cost.T * fixedprec), 
						  axis=1).astype(np.uint64)
		# Adding to graph
		g.add_weighted_edges_from(data_to_C_edges)


		a = n + K
		g.add_node(a, demand=n-np.sum(demand))
		C_to_a_edges = concatenate((array([range(n, n + K)]).T, tile([[a]], K).T), axis=1)
		g.add_edges_from(C_to_a_edges)


		# Calculating min cost flow...
		f = nx.min_cost_flow(g)

		# assign
		M_new = np.ones(n, dtype=np.int) * -1
		for i in range(n):
			p = sorted(f[i].items(), key=lambda x: x[1])[-1][0]
			M_new[i] = p - n

		# stop condition
		if np.all(M_new == M):
			# Stop
			return (C, M, f)

		M = M_new

		# compute new centers
		for k in range(K):
			C[k, :] = np.mean(data[M==k, :], axis=0)

		if maxiter is not None and itercnt >= maxiter:
			# Max iterations reached
				return (C, M, f)


def main():
	np.random.seed(824)
	data = np.random.random((75, 3))
	t = time.time()
	(C, M, f) = constrained_kmeans(data, [25, 25, 25])
	print('Elapsed:', (time.time() - t) * 1000, 'ms')
	print('C:', C)
	print('M:', M)


if __name__ == '__main__':
	main()
