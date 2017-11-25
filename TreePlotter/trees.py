# -*- coding: UTF-8 -*-
from math import log
import operator

def createDataSet():
	dataSet = [ [1,1,'yes'],
				[1,1,'yes'],
				[1,0,'no'],
				[0,1,'no'],
				[0,1,'no']]
	labels = ['no surfacing','flippers']
	return dataSet,labels

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	# 为所有可能分类创建字典
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries # 求类别出现的概率
		shannonEnt -= prob * log(prob,2)
	return shannonEnt

# 按照给定特征划分数据集 
# dataSet:待划分数据集 axis:划分数据集的特征 value:需要返回的特征值
def splitDataSet(dataSet,axis,value):
	retDataSet = []
	for featVec in dataSet:
		# 如果特征值匹配则把其他特征值提取为列表
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

# 选择最好的的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) -1
	baseEntropy = calcShannonEnt(dataSet) # 计算整个数据集的原始香农熵
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList) # 去重,创建唯一的分类标签列表
		newEntropy = 0.0
		# 计算每种划分方式的信息熵
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet,i,value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		# infoGain越大说明newEntropy越小，即按该特征分类后更稳定
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

# 如果数据集已经处理所有属性，但类标签仍然不唯一，则采用多数表决的方法决定叶子节点的分类
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=true)
	return classCount[0][0]	

# 建决策树
def createTree(dataSet,labels):
	# classList包含数据集的所有类标签
	classList = [example[-1] for example in dataSet]
	# 递归停止条件1：所有的类标签相同
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 递归停止条件2：使用完所有特征仍不能将数据集划分为仅包含唯一类别的分组
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}} # 建树，key为label，value为子树
	del(labels[bestFeat])
	featValue = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValue)
	# 对每个选中的属性中的属性值，进行递归分类
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat,value),subLabels)
	return myTree

# 使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__=='dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else: classLabel = secondDict[key]
	return classLabel

# 使用pickle序列化对象，存储决策树
def storeTree(inputTree,filename):
	import pickle
	fw = open(filename,'w')
	pickle.dump(inputTree,fw)
	fw.close()

# 从磁盘提取决策树
def grabTree(filename):
	import pickle
	fr = open(filename)
	return pickle.load(fr)
