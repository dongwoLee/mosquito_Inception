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
	L1 = tf.layers.dropout(inputs=W1,rate=0.7,training=True)

	W2_1 = tf.layers.max_pooling2d(inputs=X,pool_size=[3,3],strides=[2,2],padding="VALID")
	L2_1 = tf.layers.dropout(inputs=W2_1,rate=0.7,training=True)

	W2_2 = tf.layers.conv2d(inputs=X,filters=48,kernel_size=[3,3],strides=[2,2],padding="VALID",activation=tf.nn.relu)
	L2_2 = tf.layers.dropout(inputs=W2_2,rate=0.7,training=True)

	W2_3 = tf.layers.conv2d(inputs=W1,filters=28,kernel_size=[3,3],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_3 = tf.layers.dropout(inputs=W2_3,rate=0.7,training=True)

	W3 = tf.layers.conv2d(inputs=L2_3,filters=32,kernel_size=[3,3],strides=[2,2],padding="VALID",activation=tf.nn.relu)
	L3 = tf.layers.dropout(inputs=W3,rate=0.7,training=True)

	concat_Reduction_A = tf.concat([L2_1,L2_2,L3],axis=3)

	return concat_Reduction_A

def Inception_7B(X):

	W1_1 = tf.layers.average_pooling2d(inputs=X,pool_size=[3,3],strides=[1,1],padding="SAME")
	L1_1 = tf.layers.dropout(inputs=W1_1,rate=0.7,training=True)

	W1_2 = tf.layers.conv2d(inputs=X,filters=24,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L1_2 = tf.layers.dropout(inputs=W1_2,rate=0.7,training=True)

	W1_3 = tf.layers.conv2d(inputs=X,filters=24,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L1_3 = tf.layers.dropout(inputs=W1_3,rate=0.7,training=True)

	W2_1 = tf.layers.conv2d(inputs=L1_1,filters=16,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_1 = tf.layers.dropout(inputs=W2_1,rate=0.7,training=True)

	W2_2 = tf.layers.conv2d(inputs=X,filters=48,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_2 = tf.layers.dropout(inputs=W2_2,rate=0.7,training=True)

	W2_3 = tf.layers.conv2d(inputs=L1_2,filters=28,kernel_size=[1,7],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_3 = tf.layers.dropout(inputs=W2_3,rate=0.7,training=True)

	W2_4 = tf.layers.conv2d(inputs=L1_3,filters=24,kernel_size=[1,7],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_4 = tf.layers.dropout(inputs=W2_4,rate=0.7,training=True)

	W3_1 = tf.layers.conv2d(inputs=L2_3,filters=32,kernel_size=[1,7],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L3_1 = tf.layers.dropout(inputs=W3_1,rate=0.7,training=True)

	W3_2 = tf.layers.conv2d(inputs=L2_4,filters=28,kernel_size=[7,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L3_2 = tf.layers.dropout(inputs=W3_2,rate=0.7,training=True)

	W4 = tf.layers.conv2d(inputs=L3_2,filters=28,kernel_size=[1,7],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L4 = tf.layers.dropout(inputs=W4,rate=0.7,training=True)

	W5 = tf.layers.conv2d(inputs=L4,filters=32,kernel_size=[7,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L5 = tf.layers.dropout(inputs=W5,rate=0.7,training=True)

	concat_Inception_7B = tf.concat([L2_1,L2_2,L3_1,L5],axis=3)

	return concat_Inception_7B

def Reduction_B(X):

	W1_1 = tf.layers.max_pooling2d(inputs=X,pool_size=[3,3],strides=[2,2],padding="VALID")
	L1_1 = tf.layers.dropout(inputs=W1_1,rate=0.7,training=True)

	W1_2 = tf.layers.conv2d(inputs=X,filters=24,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L1_2 = tf.layers.dropout(inputs=W1_2,rate=0.7,training=True)

	W1_3 = tf.layers.conv2d(inputs=X,filters=32,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L1_3 = tf.layers.dropout(inputs=W1_3,rate=0.7,training=True)

	W2_1 = tf.layers.conv2d(inputs=L1_2,filters=24,kernel_size=[3,3],strides=[2,2],padding="VALID",activation=tf.nn.relu)
	L2_1 = tf.layers.dropout(inputs=W2_1,rate=0.7,training=True)

	W2_2 = tf.layers.conv2d(inputs=L1_3,filters=32,kernel_size=[1,7],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_2 = tf.layers.dropout(inputs=W2_2,rate=0.7,training=True)

	W3 = tf.layers.conv2d(inputs=L2_2,filters=40,kernel_size=[7,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L3 = tf.layers.dropout(inputs=W3,rate=0.7,training=True)

	W4 = tf.layers.conv2d(inputs=L3,filters=40,kernel_size=[3,3],strides=[2,2],padding="VALID",activation=tf.nn.relu)
	L4 = tf.layers.dropout(inputs=W4,rate=0.7,training=True)

	concat_Reduction_B = tf.concat([L1_1,L2_1,L4],axis=3)

	return concat_Reduction_B

def Inception_3C(X):

	W1_1 = tf.layers.average_pooling2d(inputs=X,pool_size=[3,3],strides=[1,1],padding="SAME")
	L1_1 = tf.layers.dropout(inputs=W1_1,rate=0.7,training=True)

	W1_2 = tf.layers.conv2d(inputs=X,filters=48,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L1_2 = tf.layers.dropout(inputs=W1_2,rate=0.7,training=True)

	W1_3 = tf.layers.conv2d(inputs=X,filters=48,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L1_3 = tf.layers.dropout(inputs=W1_3,rate=0.7,training=True)

	W2_1 = tf.layers.conv2d(inputs=X,filters=32,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_1 = tf.layers.dropout(inputs=W2_1,rate=0.7,training=True)

	W2_2 = tf.layers.conv2d(inputs=L1_3,filters=56,kernel_size=[1,3],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L2_2 = tf.layers.dropout(inputs=W2_2,rate=0.7,training=True)

	W3_1 = tf.layers.conv2d(inputs=L1_1,filters=32,kernel_size=[1,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L3_1 = tf.layers.dropout(inputs=W3_1,rate=0.7,training=True)

	W3_2 = tf.layers.conv2d(inputs=L1_2,filters=32,kernel_size=[1,3],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L3_2 = tf.layers.dropout(inputs=W3_2,rate=0.7,training=True)

	W3_3 = tf.layers.conv2d(inputs=L1_2,filters=32,kernel_size=[3,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L3_3 = tf.layers.dropout(inputs=W3_3,rate=0.7,training=True)

	W3_4 = tf.layers.conv2d(inputs=L2_2,filters=64,kernel_size=[3,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L3_4 = tf.layers.dropout(inputs=W3_4,rate=0.7,training=True)

	W4_1 = tf.layers.conv2d(inputs=L3_4,filters=32,kernel_size=[3,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L4_1 = tf.layers.dropout(inputs=W4_1,rate=0.7,training=True)

	W4_2 = tf.layers.conv2d(inputs=L3_4,filters=32,kernel_size=[1,3],strides=[1,1],padding="SAME",activation=tf.nn.relu)
	L4_2 = tf.layers.dropout(inputs=W4_2,rate=0.7,training=True)

	cocnat_Inception_3C = tf.concat([L3_1,L2_1,L3_2,L3_3,L4_1,L4_2],axis=3)

	return cocnat_Inception_3C

if __name__ == '__main__':

	Equality_128_Input = "/Users/dongwoo/Desktop/mosquito_cnn/WholeDataSet_Cluster/equality_128_Input/"
	Equality_128_Level = "/Users/dongwoo/Desktop/mosquito_cnn/WholeDataSet_Cluster/equality_128_Level/"

	TestInput_Data = "/Users/dongwoo/Desktop/mosquito_cnn/TestData/TestInput/"
	TestLevel_Data = "/Users/dongwoo/Desktop/mosquito_cnn/TestData/TestLevel/"

	Couple = MakeCouple(Equality_128_Input,Equality_128_Level)
	
	Humidity_Data, RainFall_Data,RainFallDay_Data,AvgTemp_Data,MaxTemp_Data,MinTemp_Data = InputSet(Couple,Equality_128_Input)
	LevelSet = MakeLabelSet(Couple,Equality_128_Level)
	
	# X_img = tf.reshape(X,[-1,30,30,1])
	X = tf.placeholder(tf.float32, shape=[None,30,30,1])
	Y = tf.placeholder(tf.float32,shape=[None,9])

	I4A= tf.placeholder(tf.float32,shape=[None,30,30,20])
	RA = tf.placeholder(tf.float32,shape=[None,30,30,48])
	I7B = tf.placeholder(tf.float32,shape=[None,14,14,128])
	RB = tf.placeholder(tf.float32,shape=[None,14,14,128])
	I3C = tf.placeholder(tf.float32,shape=[None,6,6,192])

	Stem_Data=MakeStemLayer(X)
	Inception_4a_Data = Inception_4A(I4A)
	Reduction_A_Data = Reduction_A(RA)
	Inception_7B_Data = Inception_7B(I7B)
	Reduction_B_Data = Reduction_B(RB)
	Inception_3C_Data = Inception_3C(I3C)

	W_first_softmax_temp = tf.layers.max_pooling2d(inputs=Reduction_B_Data,pool_size=[6,6],strides=[1,1])
	L_first_softmax_temp = tf.layers.dropout(inputs=W_first_softmax_temp,rate=0.7,training=True)
	#인셉션에서는 Reduction B 자체가 모델이므로. softmax적용해서cost값 정해줘야함. (단 [?,1,1,filters]) <- 형태로

	W1_After_Inception_3C_Data_AveP = tf.layers.average_pooling2d(inputs=Inception_3C_Data,pool_size=[6,6],strides=[1,1])
	L1_After_Inception_3C_Data_AveP = tf.layers.dropout(inputs=W1_After_Inception_3C_Data_AveP,rate=0.7,training=True)

	Dropout_Inception_3C_Data_AveP = tf.layers.dropout(inputs=L1_After_Inception_3C_Data_AveP,rate=0.8,training=True)

	first_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = L_first_softmax_temp,labels=Y))
	first_optimizer = tf.train.AdamOptimizer(0.001).minimize(first_cost)

	second_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Dropout_Inception_3C_Data_AveP,labels=Y))
	second_optimizer = tf.train.AdamOptimizer(0.001).minimize(second_cost)
	
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
		After_Inception_4a_Data = (sess.run(Inception_4a_Data,feed_dict={I4A:np_After_Stem_Input}))
		After_Reduction_A_Data = (sess.run(Reduction_A_Data,feed_dict={RA:After_Inception_4a_Data}))
		After_Inception_7B_Data =(sess.run(Inception_7B_Data,feed_dict={I7B:After_Reduction_A_Data}))
		After_Reduction_B_Data = (sess.run(Reduction_B_Data,feed_dict={RB:After_Inception_7B_Data}))
		After_Inception_3C_Data = (sess.run(Inception_3C_Data,feed_dict={I3C:After_Reduction_B_Data}))

		After_Inception_3C_Data = tf.convert_to_tensor(After_Inception_3C_Data)

		









		
