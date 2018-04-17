import tensorflow as tf 
import csv
import numpy as np
import os
import itertools
from numpy import array


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

def MakeBatch(DataList,ListLength,BatchSize):

    Output = []
    for i in range(0,ListLength,BatchSize):
        Output.append(DataList[i:i+BatchSize])
    Output = array(Output)

    return Output

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

	Equality_128_Input = "/Users/leedongwoo/Desktop/mosquito_cnn_real/WholeDataSet_Cluster/equality_128_Input/"
	Equality_128_Level = "/Users/leedongwoo/Desktop/mosquito_cnn_real/WholeDataSet_Cluster/equality_128_Level/"

	TestInput_Data = "/Users/leedongwoo/Desktop/mosquito_cnn_real/TestData/TestInput/"
	TestLevel_Data = "/Users/leedongwoo/Desktop/mosquito_cnn_real/TestData/TestLevel/"

	Couple = MakeCouple(Equality_128_Input,Equality_128_Level)
	
	Humidity_Data, RainFall_Data,RainFallDay_Data,AvgTemp_Data,MaxTemp_Data,MinTemp_Data = InputSet(Couple,Equality_128_Input)
	LevelSet = MakeLabelSet(Couple,Equality_128_Level)

	batch_size = 10
	total_batch = 100

	Batch_Humidity = MakeBatch(Humidity_Data,1000,batch_size)
	print(Batch_Humidity[0].shape)
	Batch_RainFall = MakeBatch(RainFall_Data,1000,batch_size)
	Batch_RainFallDay = MakeBatch(RainFallDay_Data,1000,batch_size)
	Batch_AvgTemp = MakeBatch(AvgTemp_Data,1000,batch_size)
	Batch_MaxTemp = MakeBatch(MaxTemp_Data,1000,batch_size)
	Batch_MinTemp = MakeBatch(MinTemp_Data,1000,batch_size)

	Batch_Level = MakeBatch(LevelSet,1000,batch_size)
	
	# X_img = tf.reshape(X,[-1,30,30,1])
	X1 = tf.placeholder(tf.float32, shape=[None,30,30,1])
	# X1_img = tf.reshape(X1,[-1,30,30,1])
	X2 = tf.placeholder(tf.float32, shape=[None,30,30,1])
	# X2_img = tf.reshape(X2,[-1,30,30,1])
	X3 = tf.placeholder(tf.float32, shape=[None,30,30,1])
	# X3_img = tf.reshape(X3,[-1,30,30,1])
	X4 = tf.placeholder(tf.float32, shape=[None,30,30,1])
	# X4_img = tf.reshape(X4,[-1,30,30,1])
	X5 = tf.placeholder(tf.float32, shape=[None,30,30,1])
	# X5_img = tf.reshape(X5,[-1,30,30,1])
	X6 = tf.placeholder(tf.float32, shape=[None,30,30,1])
	# X6_img = tf.reshape(X6,[-1,30,30,1])
	Y = tf.placeholder(tf.float32,shape=[None,9])

	keep_prob = tf.placeholder(tf.float32)

	Stem_Data_humidity = MakeStemLayer(X1) #stem data 6개
	Stem_Data_RainFall = MakeStemLayer(X2)
	Stem_Data_RainFallDay = MakeStemLayer(X3)
	Stem_Data_AvgTemp = MakeStemLayer(X4)
	Stem_Data_MaxTemp = MakeStemLayer(X5)
	Stem_Data_MinTemp = MakeStemLayer(X6)

	All_Stem_Concat_axis2 = tf.concat([Stem_Data_humidity,Stem_Data_RainFall,Stem_Data_RainFallDay,Stem_Data_AvgTemp,Stem_Data_MaxTemp,Stem_Data_MinTemp],axis=2)
	After_Stem_Input = tf.concat([All_Stem_Concat_axis2,All_Stem_Concat_axis2,All_Stem_Concat_axis2,All_Stem_Concat_axis2,All_Stem_Concat_axis2,All_Stem_Concat_axis2],axis=1)

	Inception_4a_Data = Inception_4A(After_Stem_Input)
	Reduction_A_Data = Reduction_A(Inception_4a_Data)
	Inception_7B_Data = Inception_7B(Reduction_A_Data)
	Reduction_B_Data = Reduction_B(Inception_7B_Data)
	Inception_3C_Data = Inception_3C(Reduction_B_Data)

# 	#인셉션에서는 Reduction B 자체가 모델이므로. softmax적용해서cost값 정해줘야함. (단 [?,1,1,filters]) <- 형태로
	W_first_softmax_temp = tf.layers.max_pooling2d(inputs=Reduction_B_Data,pool_size=[6,6],strides=[1,1])
	L_first_softmax_temp = tf.layers.dropout(inputs=W_first_softmax_temp,rate=0.7,training=True)

	L_first_softmax_temp = tf.reshape(L_first_softmax_temp,[-1,1*1*192])

	W1 = tf.get_variable("W1",shape=[192,96],initializer=tf.contrib.layers.xavier_initializer())
	b1 = tf.Variable(tf.random_normal([96]))
	L1 = tf.nn.relu(tf.matmul(L_first_softmax_temp,W1)+b1)
	L1 = tf.nn.dropout(L1, keep_prob=0.7)

	W2 = tf.get_variable("W2",shape=[96,48],initializer=tf.contrib.layers.xavier_initializer())
	b2 = tf.Variable(tf.random_normal([48]))
	L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
	L2 = tf.nn.dropout(L2, keep_prob=0.7)

	W3 = tf.get_variable("W3", shape=[48, 9],initializer=tf.contrib.layers.xavier_initializer())
	b3 = tf.Variable(tf.random_normal([9]))
	model_first = tf.matmul(L2, W3) + b3

	first_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = model_first,labels=Y))

	W1_After_Inception_3C_Data_AveP = tf.layers.average_pooling2d(inputs=Inception_3C_Data,pool_size=[6,6],strides=[1,1])
	L1_After_Inception_3C_Data_AveP = tf.layers.dropout(inputs=W1_After_Inception_3C_Data_AveP,rate=0.7,training=True)

	Dropout_Inception_3C_Data_AveP = tf.layers.dropout(inputs=L1_After_Inception_3C_Data_AveP,rate=0.8,training=True)

	Dropout_Inception_3C_Data_AveP = tf.reshape(Dropout_Inception_3C_Data_AveP,[-1,1*1*192])

	W4 = tf.get_variable("W4",shape=[192,96],initializer=tf.contrib.layers.xavier_initializer())
	b4 = tf.Variable(tf.random_normal([96]))
	L4 = tf.nn.relu(tf.matmul(Dropout_Inception_3C_Data_AveP,W4)+b4)
	L4 = tf.nn.dropout(L4, keep_prob=0.7)

	W5 = tf.get_variable("W5",shape=[96,48],initializer=tf.contrib.layers.xavier_initializer())
	b5 = tf.Variable(tf.random_normal([48]))
	L5 = tf.nn.relu(tf.matmul(L4,W5)+b5)
	L5 = tf.nn.dropout(L5, keep_prob=0.7)

	W6 = tf.get_variable("W6", shape=[48, 9],initializer=tf.contrib.layers.xavier_initializer())
	b6 = tf.Variable(tf.random_normal([9]))
	model_second = tf.matmul(L5, W6) + b6

	second_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_second,labels=Y))

	cost_result = 0.7*second_cost+0.3*first_cost
	optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost_result)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	for epoch in range(10):

		total_cost = 0

		for i in range(total_batch):
			
			Batch_Humidity = Batch_Humidity[i]
			Batch_Humidity = Batch_Humidity.reshape(-1,30,30,1)
			Batch_RainFall = Batch_RainFall[i]
			Batch_RainFall = Batch_RainFall.reshape(-1,30,30,1)
			Batch_RainFallDay = Batch_RainFallDay[i]
			Batch_RainFallDay = Batch_RainFallDay.reshape(-1,30,30,1)
			Batch_AvgTemp = Batch_AvgTemp[i]
			Batch_AvgTemp = Batch_AvgTemp.reshape(-1,30,30,1)
			Batch_MaxTemp = Batch_MaxTemp[i]
			Batch_MaxTemp = Batch_MaxTemp.reshape(-1,30,30,1)
			Batch_MinTemp = Batch_MinTemp[i]
			Batch_MinTemp = Batch_MinTemp.reshape(-1,30,30,1)
			
			Batch_Level = Batch_Level[i] 

			_, cost_val = sess.run([optimizer,cost_result],feed_dict={X1:Batch_Humidity,X2:Batch_RainFall,X3:Batch_RainFallDay,X4:Batch_AvgTemp,X5:Batch_MaxTemp,X6:Batch_MinTemp,
																		Y:Batch_Level,keep_prob:0.5})

			total_cost += cost_val

		print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.9f}'.format(total_cost / total_batch))

	print("완료")
	
		









		
