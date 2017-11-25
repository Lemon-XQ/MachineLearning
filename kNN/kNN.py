# -*- coding: UTF-8 -*-  

from numpy import *
import operator
from os import listdir

def createDataSet():
	group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1],[2.0,3.0],[2.0,3.3]])
	labels = ['A','A','B','B','C','C']
	return group,labels

# inX:输入向量 dataSet:输入的训练样本集 labels:标签向量 k:选择最近邻居的数目
def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0]# shape[0]表示行数
	diffMat = tile(inX,(dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndices = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndices[i]]
		print i, "voteIlabel:",voteIlabel," sortI:",sortedDistIndices[i],"distance:",distances[i]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(),
		key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	arrayOfLines = fr.readlines()
	numberOfLines = len(arrayOfLines)
	returnMat = zeros((numberOfLines,3))
	classLabelVector = [] # label list
	index = 0
	for line in arrayOfLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

# 归一化特征值 newValue = (oldValue-min)/(max-min)
def autoNorm(dataSet):
	minVals = dataSet.min(0) # 0表示从列中选取最小值而不是行
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))# 与dataSet一样维度大小
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals,(m,1))# 用最小值填充矩阵，再求差
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals

def datingClassTest():
	hoRatio = 0.10# 10%作为测试集
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	# 测试分类效果
	for i in range(numTestVecs):
		# inX:dataSet前10%  dataSet:dataSet剩余90%
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
		print "The classifier came back with: %d, the real answer is %d"\
			% (classifierResult,datingLabels[i])
		if classifierResult != datingLabels[i]:
			errorCount += 1.0
	print "Total Error Count:",errorCount
	print "Total Error Rate:%f" % (errorCount/float(numTestVecs))

# 根据三个特征判断喜好
def classifyPerson():
	resultList = ['not at all','in small doses','in large doses']
	percentTats = float(raw_input("percentage of time spent playing video games?"))
	ffMiles = float(raw_input("frequent filer miles earned per year?"))
	iceCream = float(raw_input("liters of ice cream consumed per year?"))
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
	print "You will probably like this person:",resultList[classifierResult - 1]

# 图像转向量
def img2Vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

# 手写数字识别
def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits') # 文件名格式：类别_编号.txt
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2Vector('trainingDigits/%s' % fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2Vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print "The classifier came back with: %d, the real answer is: %d" % (classifierResult,classNumStr)
		if(classifierResult != classNumStr):
			errorCount += 1.0
	print "Error Count: ",errorCount
	print "Error Rate: %f" % (errorCount/float(mTest))
