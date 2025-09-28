# 这是一个简单的决策树实现,简单在以下方面:
# 1. 训练数据分量的取值只能是1或0
# 2. 只能做二分类

import numpy as np

class DecisionTree:
	def __init__(self,num):
		self.num = num
		self.leftbranch = None
		self.rightbranch = None

	def show(self,indent=''):
		'''
		打印决策思路
		'''
		if self.leftbranch is None and self.rightbranch is None:
			print(f'predict {self.num}')
		else:
			print(f'if feature {self.num} == 1?')
			print(indent + '  ' + 'yes, ',end='')
			self.rightbranch.show(indent=indent+'    ')
			print(indent + '  ' + 'no, ',end='')
			self.leftbranch.show(indent=indent+'  ')

def entropy(p):
	if p == 0 or p == 1:
		return 0
	return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

class decision_tree_classifier():
	def __init__(self):
		self.decision_tree: DecisionTree = None

	def classify(self,train_data,labels,feature:int):
		'''
		按照给定的feature二分类
		'''
		identify_mask = (train_data[:,feature] == 1)
		subset_0 = train_data[~identify_mask]
		subset_0_labels = labels[~identify_mask]
		subset_1 = train_data[identify_mask]
		subset_1_labels = labels[identify_mask]

		return subset_0,subset_0_labels,subset_1,subset_1_labels

	def caculate_info_gain(self,train_data,labels,s0,sl0,s1,sl1):
		m = train_data.shape[0]
		m0 = s0.shape[0]
		m1 = s1.shape[0]

		if m0 == 0 or m1 == 0:
			return 0

		p = labels[labels==1].shape[0]/m
		p0 = sl0[sl0==1].shape[0]/m0
		p1 = sl1[sl1==1].shape[0]/m1

		H = entropy(p)
		H0 = entropy(p0)
		H1 = entropy(p1)

		return H - ((m0/m) * H0 + (m1/m) * H1)

	def build(self,train_data,labels):
		best_feature = 0
		info_gain = 0
		for feature in range(train_data.shape[1]): # 选出最好的feature,即information gain 最大的feature
			s0,sl0,s1,sl1 = self.classify(train_data,labels,feature) # 二分类
			new_info_gain = self.caculate_info_gain(train_data,labels,s0,sl0,s1,sl1)
			if new_info_gain > info_gain:
				info_gain = new_info_gain
				best_feature = feature

		s0,sl0,s1,sl1 = self.classify(train_data,labels,best_feature)
		if info_gain < 1e-3 or (s0.shape[0] < 2 and s1.shape[0] < 2): 
			label = 1 if np.sum(labels) >= len(labels)/2 else 0
			leaf = DecisionTree(label)
			return leaf

		d_tree = DecisionTree(best_feature)
		d_tree.leftbranch = self.build(s0,sl0)
		d_tree.rightbranch = self.build(s1,sl1)
		return d_tree

	def fit(self,train_data,labels):
		self.decision_tree = self.build(train_data,labels)

	def predict_one(self,x,node):
		if node.leftbranch is None and node.rightbranch is None:
			return node.num
		if x[node.num] == 0:
			return self.predict_one(x,node.leftbranch)
		else:
			return self.predict_one(x,node.rightbranch)
		
	def predict(self, X):
		return np.array([self.predict_one(x, self.decision_tree) for x in X])
		