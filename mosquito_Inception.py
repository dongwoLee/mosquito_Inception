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

def MakeStemLayer(X): # X is tf.placeholder in main function
	
	W1 = tf.layers.conv2d(inputs=X,filters=4,strides=[2,2],kernel_size=[3,3],padding="VALID",activation = tf.nn.relu)
	L1 = tf.layers.dropout(inputs=W1,rate=0.7,training = True)

	W2 = tf.layers.conv2d(inputs=L1,filters=4,strides=[1,1],kernel_size=[3,3],padding="VALID",activation=tf.nn.relu)
	L2 = tf.layers.dropout(inputs=W2,rate=0.7,training = True)

	W3 = tf.layers.conv2d(inputs=L2,filters=8,strides=[1,1],kernel_size=[3,3],padding="SAME",activation=tf.nn.relu)
	L3 = tf.layers.dropout(inputs=W3,rate=0.7,training = True)

	W4_1 = tf.layers.max_pooling2d(inputs=L3,pool_size=[3,3],strides=[2,2],padding="VALID")
	L4_1 = tf.layers.dropout(inputs=W4_1,rate=0.7,training=True)

	W4_2 = tf.layers.conv2d(inputs=L3,filters=12,kernel_size=[3,3],strides=[2,2],padding="VALID",activation=tf.nn.relu)
	L4_2 = tf.layers.dropout(inputs=W4_2,rate=0.7,training=True)

	concat_data_1 = tf.concat([L4_1,L4_2],axis=3)
	
	return (concat_data_1)

def Inception_4A(X):

	W1_1 = tf.layers.average_pooling2d(inputs=X,pool_size=[3,3],strides=[1,1],padding="SAME")
	L1_1 = tf.layers.dropout(inputs=W1_1,rate=0.7,training=True)

	W1_2 = tf.layers.conv2d(inputs=X,filters=12,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L1_2 = tf.layers.dropout(inputs=W1_2,rate=0.7,training=True)

	W1_3 = tf.layers.conv2d(inputs=X,filters=8,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L1_3 = tf.layers.dropout(inputs=W1_3,rate=0.7,training=True)

	W1_4 = tf.layers.conv2d(inputs=X,filters=8,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L1_4 = tf.layers.dropout(inputs=W1_4,rate=0.7,training=True)

	W2_1 = tf.layers.conv2d(inputs=L1_1,filters=12,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_1 = tf.layers.dropout(inputs=W2_1,rate=0.7,training=True)

	W2_2 = tf.layers.conv2d(inputs=L1_3,filters=12,kernel_size=[3,3],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_2 = tf.layers.dropout(inputs=W2_2,rate=0.7,training=True)

	W2_3 = tf.layers.conv2d(inputs=L1_4,filters=12,kernel_size=[3,3],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_3 = tf.layers.dropout(inputs=W2_3,rate=0.7,training=True)

	W3 = tf.layers.conv2d(inputs=L2_3,filters=12,kernel_size=[3,3],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L3 = tf.layers.dropout(inputs=W3,rate=0.7,training=True)

	concat_Inception_4A = tf.concat([L2_1,L1_2,L2_2,L3],axis=3)

	return concat_Inception_4A

def Reduction_A(X):

	W1 = tf.layers.conv2d(inputs=X,filters=24,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L1 = tf.layers.dropout(inputs=W1,rate=0.7,traning=True)

	W2_1 = tf.layers.max_pooling2d(inputs=X,pool_size=[3,3],strides=[2,2],padding="VALID")
	L2_1 = tf.layers.dropout(inputs=W2_1,rate=0.7,training=True)

	W2_2 = tf.layers.conv2d(inputs=X,filters=48,kernel_size=[3,3],strides=[2,2],padding="VALID",activation=tf.nn.relu)
	L2_2 = tf.layers.dropout(inputs=W2_2,rate=0.7,training=True)

	W2_3 = tf.layers.conv2d(inputs=W1,filters=28,kernel_size=[3,3],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_3 = tf.layers.dropout(inputs=W2_3,rate=0.7,training=True)

	W3 = tf.layers.conv2d(inputs=L2_3,filters=32,kernel_size=[3,3],strides=[2,2],padding="VALID",activation=tf.nn.relu)
	L3 = tf.layers.dropout(inputs=L3,rate=0.7,training=True)

	concat_Reduction_A = tf.concat([L2_1,L2_2,L3],axis=3)

	return concat_Reduction_A


if __name__ == '__main__':

	Equality_128_Input = "/Users/leedongwoo/Desktop/mosquito_cnn_real/WholeDataSet_Cluster/equality_128_Input/"
	Equality_128_Level = "/Users/leedongwoo/Desktop/mosquito_cnn_real/WholeDataSet_Cluster/equality_128_Level/"

	TestInput_Data = "/Users/leedongwoo/Desktop/mosquito_cnn_real/TestData/TestInput/"
	TestLevel_Data = "/Users/leedongwoo/Desktop/mosquito_cnn_real/TestData/TestLevel/"

	Couple = MakeCouple(Equality_128_Input,Equality_128_Level)
	
	Humidity_Data, RainFall_Data,RainFallDay_Data,AvgTemp_Data,MaxTemp_Data,MinTemp_Data = InputSet(Couple,Equality_128_Input)
	LevelSet = MakeLabelSet(Couple,Equality_128_Level)
	
	# X_img = tf.reshape(X,[-1,30,30,1])
	X = tf.placeholder(tf.float32, shape=[None,30,30,1])
	Y = tf.placeholder(tf.float32,shape=[None,9])

	I4A= tf.placeholder(tf.float32,shape=[None,30,30,20])
	RA = tf.placeholder(tf.float32,shape=[None,30,30,48])

	Stem_Data=MakeStemLayer(X)
	Inception_4a_Data = Inception_4A(I4A)
	Reduction_A_Data = Reduction_A(RA)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		humidity_stem = (sess.run(Stem_Data, feed_dict={X: Humidity_Data[0].reshape(-1,30,30,1)}))
		RainFall_stem= (sess.run(Stem_Data, feed_dict={X: RainFall_Data[0].reshape(-1,30,30,1)}))
		RainFallDay_stem = (sess.run(Stem_Data, feed_dict={X: RainFallDay_Data[0].reshape(-1,30,30,1)}))
		AvgTemp_stem = (sess.run(Stem_Data, feed_dict={X: AvgTemp_Data[0].reshape(-1,30,30,1)}))
		MaxTemp_stem = (sess.run(Stem_Data, feed_dict={X: MaxTemp_Data[0].reshape(-1,30,30,1)}))
		MinTemp_stem = (sess.run(Stem_Data, feed_dict={X: MinTemp_Data[0].reshape(-1,30,30,1)}))

		All_Stem_Concat_axis2= tf.concat([humidity_stem,RainFall_stem,RainFallDay_stem,AvgTemp_stem,MaxTemp_stem,MinTemp_stem],axis=2)
		After_Stem_Input = tf.concat([All_Stem_Concat_axis2,All_Stem_Concat_axis2,All_Stem_Concat_axis2,
										All_Stem_Concat_axis2,All_Stem_Concat_axis2,All_Stem_Concat_axis2],axis=1)
		# print(After_Stem_Input.shape)
		np_After_Stem_Input = After_Stem_Input.eval()
		Inception_4a_Data = (sess.run(Inception_4a_Data,feed_dict={I4A:np_After_Stem_Input}))
		