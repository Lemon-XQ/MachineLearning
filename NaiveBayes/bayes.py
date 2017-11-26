# -*- coding:utf-8 -*-

# 以在线社区留言板为例，用朴素贝叶斯分类器对文本进行分类：侮辱类1 非侮辱类0
from numpy import *
import feedparser

#--------------------------------------------------
# 词表到向量的转换函数

def loadDataSet():
	postingList = [['my','dog','has','flea','problems','help','please'],
	               ['maybe','not','take','him','to','dog','park','stupid'],
	               ['my','dalmation','is','so','cute','I','love','him'],
	               ['stop','posting','stupid','worthless','garbage'],
	               ['mr','licks','ate','my','steak','how','to','stop','him'],
	               ['quit','buying','worthless','dog','food','stupid']]
	classVec = [0,1,0,1,0,1] # 侮辱类1 非侮辱类0
	return postingList, classVec

# 创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document) # 创建两个集合的并集
	return list(vocabSet)

# 词集模型
# 输入：词汇表 + 某个文档	输出：文档向量 0表示词汇表中的单词在输入文档中没出现 1表示出现
def setOfWords2Vec(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1 # 词集模型
		else:
			print "Word: %s is not in Vocabulary" % word
	return returnVec

# 词袋模型
# 输入：词汇表 + 某个文档	输出：文档向量 单词出现次数
def bagOfWords2Vec(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1 # 词袋模型，每个单词可出现多次
		else:
			print "Word: %s is not in Vocabulary" % word
	return returnVec

#------------------------------------------------
# 训练函数：从词向量计算条件概率 p(ci|w) = p(w|ci)p(ci) / p(w)
# p(ci): 类别i中文档数/总文档数
# p(w|ci): 条件独立性假设下等于p(w0|ci)p(w1|ci)...p(wn|ci)

# 输入：文档矩阵 + 由每篇文档类别标签所构成的向量
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix) # 文档总数
	numWords = len(trainMatrix[0]) # 词汇总数，即词汇表长度
	print numTrainDocs,numWords
	pAbusive = sum(trainCategory)/float(numTrainDocs) # 计算文档属于侮辱性文档的概率
	# 分子(只要1个概率为0最后乘积为0，所以为加快计算将所有词出现数初始化为1，总词数初始化为2)
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	# 分母
	p0Denom = 2.0
	p1Denom = 2.0
	# 遍历每个训练文档，分别统计侮辱类和非侮辱类中各词汇的出现次数
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i]) # 文档i中的总词数
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	# 取自然对数避免下溢出
	p1Vect = log(p1Num/p1Denom) # 侮辱类中各单词出现概率
	p0Vect = log(p0Num/p0Denom) # 非侮辱类中各单词出现概率
	return p0Vect,p1Vect,pAbusive

#------------------------------------------------------
# 测试算法

# 输入：文档向量，非/侮辱类单词出现概率的向量，文档属于侮辱性概率p(ci)
# 输出：分类结果
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1) # p1 = p1Num*pClass1 /p1Denom
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if(p1 > p0):
		return 1
	else:
		return 0

def testNB():
	listOfPosts,listClasses = loadDataSet()
	vocabList = createVocabList(listOfPosts)
	trainMat = []
	for postInDoc in listOfPosts:
		trainMat.append(setOfWords2Vec(vocabList,postInDoc))
	p0V,p1V,pAb = trainNB0(trainMat, listClasses)
	testCase = ['love','my','dalmation']
	testCaseVec = setOfWords2Vec(vocabList, testCase)
	print testCase,' classifyed as: ',classifyNB(testCaseVec, p0V, p1V, pAb)
	testCase = ['stupid','garbage']
	testCaseVec = setOfWords2Vec(vocabList, testCase)
	print testCase,' classifyed as: ',classifyNB(testCaseVec, p0V, p1V, pAb)

#-------------------------------------------------------------------------
# 应用：垃圾邮件过滤

# 将文本解析为字符串列表
def textParse(text):
	import re
	listOfTokens = re.split(r'\W*', text) # 分隔符是除单词数字外的任意字符串
	return [tok.lower() for tok in listOfTokens if len(tok) > 2] # 去掉少于2个字符的字符串并将所有字符串转换为小写

def spamTest():
	docList = []; classList=[]; fullText = []
	# 导入并解析文本文件
	for i in range(1,26):
		wordList = textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	# 留存交叉验证
	trainingSet = range(50) # 0-49的整数列表
	testSet = []
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex]) # 删除该元素
		print randIndex, trainingSet[randIndex]
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))
	errorCount = 0
	# 计算分类错误率
	for docIndex in testSet:
		wordVec = setOfWords2Vec(vocabList, docList[docIndex])
		# array(wordVec)转换wordVec为一个元素 如[0,0,1]->[0 0 1]
		if classifyNB(array(wordVec), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1
	print 'ERROR RATE IS: ' ,(float(errorCount)/len(testSet))


#-------------------------------------------------------------------
# 应用：从个人广告中获取区域倾向

def calcMostFreq(vocabList,fullText):
	import operator
	freqDict = {}
	# 计算出现频率
	for token in vocabList:
		freqDict[token] = fullText.count(token)
	# freqDict数据项格式(u'and', 52)，按第二个域排序
	sortedFreq = sorted(freqDict.iteritems(),key=operator.itemgetter(1),reverse=True)
	print 'sorted:',sortedFreq[:30] 
	return sortedFreq[:30]

# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# 外部导入RSS源（RSS源会随时间改变）
def localWords(feed1,feed0):
	docList = []; classList = []; fullText = []
	minLen = min(len(feed1['entries']),len(feed0['entries']))
	for i in range(minLen):
		wordList = textParse(feed1['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	top30Words = calcMostFreq(vocabList, fullText)
	for pairW in top30Words:
		if pairW[0] in vocabList:
			print 'pairW[0] ',pairW[0]
			vocabList.remove(pairW[0]) # 去除出现次数最高的那些词
	trainingSet = range(2*minLen); testSet = []
	for i in range(20):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(trainMat, trainClasses)
	errorCount = 0
	for docIndex in testSet:
		wordVec = bagOfWords2Vec(vocabList, docList[docIndex])
		if classifyNB(wordVec,p0V,p1V,pSpam) != classList[docIndex]:
			errorCount += 1
	print "ERROR RATE IS：", float(errorCount)/len(testSet)
	return vocabList,p0V,p1V

# 最具表征性的词汇显示函数
def getTopWords(ny,sf):
	import operator
	vocabList, p0V, p1V = localWords(ny, sf)
	topNY = []; topSF = []
	for i in range(len(p0V)):
		print 'p0V',i,' ',p0V[i]
		print 'p1V',i,' ',p1V[i]
		if p0V[i] > -5.0: # log(概率)
			topSF.append((vocabList[i], p0V[i]))
		if p1V[i] > -5.0:
			topNY.append((vocabList[i], p1V[i]))
	sortedSF = sorted(topSF,key=lambda pair:pair[1], reverse=True)# 按第二个域排序
	print "----------------------SF-------------------------"
	for item in sortedSF:
		print item[0]
	sortedNY = sorted(topNY,key=lambda pair:pair[1], reverse=True)
	print "----------------------NY-------------------------"
	for item in sortedNY:
		print item[0]

