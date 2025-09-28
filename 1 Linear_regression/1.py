import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predict(x,w):
	'''
	x: 一组特征(n,)
	w: 模型的参数(n+1,)
	'''
	if(len(x) != len(w)-1):
		raise ValueError("特征与参数不匹配！")
	
	return (x*w[:len(w)-1]).sum()

def cost(x,w,y):
	'''
	x: 样本(m,n)
	w: 预测参数(n,)
	y: 真实标签(m,)
	'''
	predictions = [predict(z,w) for z in x]
	return ((predictions - y)**2).sum() * 1/(2*x.shape[0])

def gradian_descent(w,x,y,alpha,iter=1000):
	'''
	x: 样本(m,n)\n
	w: 预测参数(n,)\n
	y: 真实标签(m,)\n
	alpha: 学习率
	iter: 最大迭代次数
	'''

	temp = w
	iter_cnt = 0

	for _ in range(iter):
		iter_cnt += 1
		c = temp

		#print('temp:',temp.shape)
		#print(x[1].shape)
		predictions = [predict(z,temp) for z in x]
		errors = predictions - y

		gradiant = (1/x.shape[0])*(errors@x)
		#print('g shape:',gradiant.shape)
		gradiant = gradiant/np.linalg.norm(gradiant, ord=2)

		temp = temp-alpha*gradiant
		
		if(((c-temp)**2).sum() < 1e-6):
			break

	print(f'iter cnt: {iter_cnt}')
	return temp