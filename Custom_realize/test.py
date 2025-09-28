import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


class logistic_regression():
	def __init__(self,C=1.0):
		self.w = 0
		self.b = 0
		self.C = C#正则化参数

	def logistic(self,z):
		return 1/(1 + np.exp((-1)*z))
	
	def predict(self,x):
		#预测，x可以是(m,n)矩阵
		return self.logistic(x@self.w + self.b)
	
	def loss(self,x,y):
		'''
		计算单例的损失\n
		x: a sample (n,)\n
		y: a label
		'''
		p = self.predict(x)
		return (-1)*y * np.log(p) - (1-y)*np.log(1-p)

	def cost(self,X,y):
		'''
		成本函数\n
		X: samples (m,n)\n
		y: labels (m,)
		'''
		losses = np.array([self.loss(X[i],y[i]) for i in range(y.shape[0])])
		return (1/y.shape[0])*losses.sum()

	def fit(self,X,y,alpha,max_iter=5000):
		'''
		x: 样本(m,n)\n
		y: 真实标签(m,)\n
		alpha: 学习率\n
		iter: 最大迭代次数
		'''

		self.w = np.zeros(X.shape[1])
		self.b = 0
		iter_cnt = 0
		costs = [self.cost(X,y)]

		for _ in range(max_iter):
			iter_cnt += 1
			c = self.w
			d = self.b

			#print('temp:',temp.shape)
			#print(x[1].shape)
			difference = np.array([self.predict(X[i]) - y[i] for i in range(y.shape[0])])
			self.w = self.w * (1-alpha*self.C/X.shape[0]) - alpha*(difference@X)
			self.b = self.b - alpha*difference.sum()

			costs.append(self.cost(X,y))

			if(np.sqrt(((c-self.w)**2).sum() + (d - self.b)**2) < 1e-8):
				break

		print(f'iter cnt: {iter_cnt}')

		return costs
	
# 加载数据
X, y = load_iris(return_X_y=True)

# 只用前两类来做二分类
X, y = X[y != 2], y[y != 2]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建逻辑回归模型（C=正则化强度，越小越强）
model = LogisticRegression(solver='lbfgs')

# 拟合模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算准确率
print("准确率：", accuracy_score(y_test, y_pred))

print("权重 w:", model.coef_)
print("偏置 b:", model.intercept_)

#====================================================================

my_model = logistic_regression(C=1)

my_model.fit(X_train, y_train,0.003)

y_pred = my_model.predict(X_test)

y_pred[y_pred>0.5]=1
y_pred[y_pred<=0.5]=0

print("准确率：", accuracy_score(y_test, y_pred))

print("权重 w:", my_model.w)
print("偏置 b:", my_model.b)

#为何与scikit-learn相差这么多？虽然我的准确率尚可