import numpy as np

class KMeans():
	def __init__(self):
		self.data = None
		self.idx = None
		self.centroids = None
	
	def find_closest_centroids(self, X, centroids):
			"""
			Computes the centroid memberships for every example
			
			Args:
					X (ndarray): (m, n) Input values      
					centroids (ndarray): k centroids
			
			Returns:
					idx (array_like): (m,) closest centroids
			
			"""
			# Set K
			K = centroids.shape[0]

			idx = np.zeros(X.shape[0], dtype=int)

			for num,x in enumerate(X):
					idx[num] = np.argmin(np.linalg.norm(centroids - x,axis=1))
			
			return idx

	def compute_centroids(self, X, idx, K):
			"""
			Returns the new centroids by computing the means of the 
			data points assigned to each centroid.
			
			Args:
					X (ndarray):   (m, n) Data points
					idx (ndarray): (m,) Array containing index of closest centroid for each 
												example in X. Concretely, idx[i] contains the index of 
												the centroid closest to example i
					K (int):       number of centroids
			
			Returns:
					centroids (ndarray): (K, n) New centroids computed
			"""
			m, n = X.shape
			
			centroids = np.zeros((K, n))
			
			cnts = np.zeros((K,1))
			for num,x in enumerate(X):
					cnts[idx[num]] += np.array([1])
					centroids[idx[num]] += x
			centroids = centroids/cnts
			
			return centroids
	
	def init_centroids(self, X, K):
			"""
			This function initializes K centroids that are to be 
			used in K-Means on the dataset X
			
			Args:
					X (ndarray): Data points 
					K (int):     number of centroids/clusters
			
			Returns:
					centroids (ndarray): Initialized centroids
			"""
			
			# Randomly reorder the indices of examples
			randidx = np.random.permutation(X.shape[0])
			
			# Take the first K examples as centroids
			centroids = X[randidx[:K]]
			
			return centroids
	
	def run(self, X, K, max_iters=10):
			"""
			Runs the K-Means algorithm on data matrix X, where each row of X
			is a single example
			"""
			self.data = X
			initial_centroids = self.init_centroids(X,K)
			m, n = X.shape
			centroids = initial_centroids
			previous_centroids = centroids    
			idx = np.zeros(m)
			
			# Run K-Means
			for i in range(max_iters):
					#Output progress
					print("K-Means iteration %d/%d" % (i, max_iters-1))

					# For each example in X, assign it to the closest centroid
					self.idx = self.find_closest_centroids(X, centroids)
							
					# Given the memberships, compute new centroids
					self.centroids = self.compute_centroids(X, idx, K)
			return self.centroids, self.idx