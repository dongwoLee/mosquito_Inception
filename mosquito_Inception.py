import tensorflow as tf 
import csv
import numpy as np
import os
import itertools

def MakeCouple(TrainInputDir,TrainLevelDir):

	TrainInput_CsvList = os.listdir(TrainInputDir)
	TrainInput_CsvList.sort()
	TrainLevel_CsvList = os.listdir(TrainLevelDir)
	TrainLevel_CsvList.sort()

	CoupleList = []
	TrainLength = len(TrainInput_CsvList) # List length

	for i in range(TrainLength):
		temp =[]
		temp.append([TrainInput_CsvList[i],TrainLevel_CsvList[i]])
		CoupleList.extend(temp)

	return CoupleList

#————————————————about Input——————#

def InputSet(RawList,InputDir):

	WholeList = []
	Humidity_All,RainFall_All,RainFallDay_All,AvgTemperature_All,MaxTemperature_All,MinTemperature_All=[],[],[],[],[],[]
	
	for i in range(len(RawList)):
		with open(InputDir+RawList[i][0],'r') as csvfile:
			reader = csv.reader(csvfile)
			WholeList = list(reader)
				
			Humidity = WholeList[0:30]
			Humidity_All.extend(Humidity)
			RainFall = WholeList[30:60]
			RainFall_All.extend(RainFall)
			RainFallDay = WholeList[60:90]
			RainFallDay_All.extend(RainFallDay)
			AvgTemperature = WholeList[90:120]
			AvgTemperature_All.extend(AvgTemperature)
			MaxTemperature = WholeList[120:150]
			MaxTemperature_All.extend(MaxTemperature)
			MinTemperature = WholeList[150:180]
			MinTemperature_All.extend(MinTemperature)

	Humidity_All = UnfoldList_SplitBy900(Humidity_All)
	RainFall_All = UnfoldList_SplitBy900(RainFall_All)
	RainFallDay_All = UnfoldList_SplitBy900(RainFallDay_All)
	AvgTemperature_All = UnfoldList_SplitBy900(AvgTemperature_All)
	MaxTemperature_All = UnfoldList_SplitBy900(MaxTemperature_All)
	MinTemperature_All = UnfoldList_SplitBy900(MinTemperature_All)

	return (Humidity_All),(RainFall_All),(RainFallDay_All),(AvgTemperature_All),(MaxTemperature_All),(MinTemperature_All)


def UnfoldList_SplitBy900(FactorList):

	temp=[]
	for i in range(len(FactorList)):
		temp += FactorList[i]
	
	temp = list(map(float,temp))
	temp_array = np.array([temp[k:k+900] for k in range(0,len(temp),900)])
	
	return temp_array

#———————————————————about LevelSet——————————

def MakeLabelSet(RawLevelList,LevelDir):

	WholeLevelList = []
	depth = 9
	for i in range(len(RawLevelList)):
		with open(LevelDir+RawLevelList[i][1],'r') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if(row==[]):
					continue
				else:
					WholeLevelList.extend(row)

	WholeLevelList = list(map(int,WholeLevelList))
	tmp_one_hot  = tf.one_hot(WholeLevelList,depth)

	with tf.Session() as sess:
		one_hot_LevelList = sess.run(tmp_one_hot)

	return (one_hot_LevelList)

#-----------------Make Batch Size ---------

def MakeBatch(MakeBatchList,length):

	return np.array([MakeBatchList[i:i+length] for i in range(0,len(MakeBatchList),length)])

#-----------------Make Stem Layer ---------

def MakeStemLayer(x,W):

	W1 = tf.random_normal() 
	L1 = tf.conv2d(x,W,strides=[1,2,2,1],PADDING="VALID")

if __name__ == '__main__':

	Equality_128_Input = "/Users/leedongwoo/Desktop/mosquito_cnn_real/WholeDataSet_Cluster/equality_128_Input/"
	Equality_128_Level = "/Users/leedongwoo/Desktop/mosquito_cnn_real/WholeDataSet_Cluster/equality_128_Level/"

	TestInput_Data = "/Users/leedongwoo/Desktop/mosquito_cnn_real/TestData/TestInput/"
	TestLevel_Data = "/Users/leedongwoo/Desktop/mosquito_cnn_real/TestData/TestLevel/"

	Couple = MakeCouple(Equality_128_Input,Equality_128_Level)
	
	Humidity_Data, RainFall_Data,RainFallDay_Data,AvgTemp_Data,MaxTemp_Data,MinTemp_Data = InputSet(Couple,Equality_128_Input)
	LevelSet = MakeLabelSet(Couple,Equality_128_Level)

	X = tf.placeholder(tf.float32, shape=[None, 900])
    Y = tf.placeholder(tf.float32, shape=[None, 9])



	

	







