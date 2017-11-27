# -*- coding:utf-8 -*-

from numpy import *
#-----------------------------------------------
# Logistic回归梯度上升优化算法

def loadDataSet():
	dataMat = []; labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) # X1X2是两个数值型特征，将X0的值设为1
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat

def sigmoid(inX):
	return 1.0/(1+exp(-inX))

# 输入： dataMat二维numpy数组,100*3 100个样本3个特征
#		labelMat类别标签 1*100
def gradAscent(dataMat,labelMat):
	dataMatrix = mat(dataMat)
	labelMatrix = mat(labelMat).transpose()
	m,n = shape(dataMatrix) # m行n列
	alpha = 0.001 # 步长
	maxCycles = 500 # 迭代次数
	weights = ones((n,1)) # n个1
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)
		error = (labelMatrix - h) # 计算真实类别与预测类别的差值，按照该差值的方向调整回归系数
		weights = weights + alpha * dataMatrix.transpose() * error # 梯度上升
	return weights

#getA()方法，其将matrix()格式矩阵转为array()格式，type(weights),type(weights.getA())可观察到。  
#plotBestFit(weights.getA())  
def plotBestFit(weights):
	import matplotlib.pyplot as plt
	dataMat,labelMat = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='green')
	x = arange(-3.0, 3.0, 0.1) # 以0.1为步长，-3.0~3.0的数组
	y = (-weights[0]-weights[1]*x)/weights[2] #拟合曲线为0 = w0*x0+w1*x1+w2*x2, 故x2 = (-w0*x0-w1*x1)/w2, x0为1,x1为x, x2为y 
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()

# 随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
	m,n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights

# 改进的随机梯度上升算法，计算回归系数向量
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
	m,n = shape(dataMatrix)
	weights = ones(n)
	# j是迭代次数，i是样本点的下标
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			# 当j远小于i时alpha不是严格下降——模拟退火
			alpha = 0.01 + 4/(1.0+j+i) # 每次迭代alpha都会减小,但不会减为0，保证多次迭代后新数据仍有一定影响
			randIndex = int(random.uniform(0,len(dataIndex))) # 随机选取样本来更新回归系数，减少周期性的波动
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex]) # 每次均选取不一样的随机数
	return weights

#--------------------------------------------------------------
# 应用：预测病马的死亡率

def classifyVector(inX,weights):
	prob = sigmoid(sum(inX*weights))
	if prob > 0.5:
		return 1.0
	else:
		return 0.0

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		curLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(curLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(curLine[21]))
	trainWeights = stocGradAscent1(array(trainingSet), trainingLabels,500)
	errorCount = 0
	numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		curLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(curLine[i]))
		if int(classifyVector(array(lineArr),trainWeights)) != int(curLine[21]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec)
	print 'ERROR RATE IS:',errorRate
	return errorRate

# 调用10次colicTest并求结果的平均值
def multiTest():
	numTests = 10; errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print 'After %d iterations the ERROR RATE is: %f' % (numTests,errorSum/float(numTests))


