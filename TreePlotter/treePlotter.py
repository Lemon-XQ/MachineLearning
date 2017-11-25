# -*- coding:UTF-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt

# 中文显示
# mpl.rcParams['font.sans-serif'] = ['AR PL UKai CN']
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/YaHei Consolas Hybrid/YaHei Consolas Hybrid 1.12.ttf')

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8") # fc表示透明度
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# 绘制带箭头的注解
# nodeText:节点文字 centerPt:节点中心位置 parentPt:箭头终点位置
def plotNode(nodeText, centerPt, parentPt, nodeType):
	createPlot.ax1.annotate(nodeText,\
		xy=parentPt,xycoords='axes fraction',\
		xytext=centerPt,textcoords='axes fraction',\
		va="center", ha="center",bbox=nodeType,arrowprops=arrow_args,fontproperties=zhfont)

def createPlot(inTree):
	fig = plt.figure(1,facecolor='white') # 创建一个新图形
	fig.clf() # 清空绘图区
	axprops = dict(xticks=[],yticks=[])
	createPlot.ax1 = plt.subplot(111,frameon=False,**axprops) # 111表示分成1行1列，子图位于第一个位置
	# 全局变量 分别为树的宽度，深度，已经绘制的节点位置以及下一个节点的恰当位置
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
	plotTree.xOff = -0.5/plotTree.totalW
	plotTree.yOff = 1.0
	plotTree(inTree, (0.5,1.0), '')
	plt.show()

# 获取叶节点的数目
def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		# 如果value仍然为dict，说明还没到叶子节点
		if type(secondDict[key]).__name__=='dict':
			numLeafs += getNumLeafs(secondDict[key]) # 递归
		else:
			numLeafs += 1
	return numLeafs

# 获取树的层数
def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth = 1 + getTreeDepth(secondDict[key]) # 递归
		else:
			thisDepth = 1
		if thisDepth > maxDepth: maxDepth = thisDepth
	return maxDepth

def retriveTree(i):
	listOfTrees = [{'no surfacing':{0:'no', 1:{'flippers': {0: 'no', 1:'yes'}}}},
				   {'no surfacing':{0:'no', 1:{'flippers': {0: {'head':{0:'no',1:'yes'}}, 1:'no'}}}}]
	return listOfTrees[i]

# 在父子节点之间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeText):
	numLeafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstStr = myTree.keys()[0]
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
	plotMidText(cntrPt, parentPt, nodeText)
	plotNode(firstStr, cntrPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			plotTree(secondDict[key],cntrPt,str(key)) # 递归
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff,plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff,plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
