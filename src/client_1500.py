import socket
import json
from pyspark.context import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
#from sklearn.feature_extraction.text import CountVectorizer
from pyspark.sql.functions import lower, col, udf, regexp_replace
import re
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import HashingVectorizer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
import time
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

# Initialize the spark context.
sc = SparkContext(appName="SSML")
ssc = StreamingContext(sc, 5)

spark = SparkSession(sc)
schema = StructType([StructField("feature0", StringType(), True), StructField("feature1", StringType(), True), StructField("feature2", StringType(), True)])

global n
n = 1

mdl1 = list()
mdl2 = list()
mdl3 = list()
mdl4 = list()

accuracies = list()
precisions = list()
recalls = list()


clf1 = MultinomialNB()
clf2 = SGDClassifier()
clf3 = BernoulliNB()

kmModel = MiniBatchKMeans(n_clusters=2)

label_encoder = preprocessing.LabelEncoder()
HashVec = HashingVectorizer(alternate_sign=False)


def removeall(string):
	stopwords = text.ENGLISH_STOP_WORDS
	stopwords = list(stopwords)
	stopwords.append('\n')
	stopwords.append('\t')
	remove = ["!", "(", ")", "-", "[", "]", "{", "}", ";", ":", "'", "\"", "\\", "<", ">", ".", ",", "/", "?", "@", '#', "$", "%", "^", "&", "*", "_", "~", "1", "2", "3", "4", "5", "6" ,"7", "8", "9", "0"]
		
	toremove = stopwords + remove
	
	tokeep = [word for word in string.split() if word.lower() not in toremove]
	new_text = " ".join(tokeep)
	
	return new_text

def mbkmeans_clusters(X, k, mb, print_silhouette_values) :
	
	km = kmModel.partial_fit(X)
	print("For n_clusters = ", k)
	print(f"Silhouette coefficient : {silhouette_score(X,km.labels_):0.2f}")
	print(f"Inertia:{km.inertia_}")
				
	return km, km.labels_

#lines: type rdd
def df_fun(lines):
	exists = len(lines.collect())
	if exists:
		global n
		################################################ PREPROCESSING ################################################
		
		df = spark.createDataFrame(json.loads(lines.collect()[0]).values(), schema)
		removed0 = udf(removeall, StringType())
		new_df = df.withColumn("feature1", removed0(df["feature1"]))
		
		removed1 = udf(removeall, StringType())
		new_df1 = new_df.withColumn("feature0", removed1(new_df["feature0"]))
		
		#################################################################################################################
	
		le = label_encoder.fit_transform(np.array([row["feature2"] for row in new_df1.collect()]))

		re_data = new_df1.collect()

		X = HashVec.fit_transform([" ".join([row["feature0"], row["feature1"]]) for row in re_data])

		################################################ MODEL BUILDING AND TESTING ################################################

		
		#X_train, X_test, y_train, y_test = train_test_split(X, le, test_size = 0.33)
		
		X_train = X
		y_train = le
		X_test = X
		y_test = le
		
		print("#batch", n)
		
		if n <= 20:
			model1 = clf1.partial_fit(X_train, y_train, classes = np.unique(y_train))
			print("Model1 Training Done")			
			
		else:
			if n>20 and n<=22:
				pred1 = clf1.predict(X_test)
				
				ac1 = accuracy_score(y_test, pred1)
				prec1 = precision_score(y_test, pred1)
				rec1 = recall_score(y_test, pred1)
				conf_matrix1 = confusion_matrix(y_test, pred1)
				mdl1.append([ac1, prec1, rec1])

				print("------------------Model 1-----------------")
				print("Accuracy Score: ", ac1)
				print("Precision Score: ", prec1)
				print("Recall Score: ", rec1)
				print("Confusion Matrix: \n", conf_matrix1)
				
		
		################################ TRAIN ###################################
		if n <= 20:
			
			model2 = clf2.partial_fit(X_train, y_train, classes = np.unique(y_train))
			print("Model2 Training Done")

		################################ TEST #####################################
		else:	
			if n>20 and n<=22:
				pred2 = clf2.predict(X_test)
				
				ac2 = accuracy_score(y_test, pred2)
				prec2 = precision_score(y_test, pred2)
				rec2 = recall_score(y_test, pred2)
				conf_matrix2 = confusion_matrix(y_test, pred2)
				mdl2.append([ac2, prec2, rec2])
				
				print("------------------Model 2-----------------")
				print("Accuracy Score: ", ac2)
				print("Precision Score: ", prec2)
				print("Recall Score: ", rec2)
				print("Confusion Matrix: \n", conf_matrix2)
		

		if n <= 20:		
			model3 = clf3.partial_fit(X_train, y_train, classes = np.unique(y_train))
			print("Model3 Training Done")

		else:	
			if n>20 and n<=22:
				pred3 = clf3.predict(X_test)
				
				ac3 = accuracy_score(y_test, pred3)
				prec3 = precision_score(y_test, pred3)
				rec3 = recall_score(y_test, pred3)
				conf_matrix3 = confusion_matrix(y_test, pred3)
				mdl3.append([ac3, prec3, rec3])
				
				print("------------------Model 3-----------------")
				print("Accuracy Score: ", ac3)
				print("Precision Score: ", prec3)
				print("Recall Score: ", rec3)
				print("Confusion Matrix: \n", conf_matrix3)
				
		
		################################################ PERFORMANCE METRICS #################################################

		
		n += 1

		if n>22:
			#print("Model 1: ", mdl1)
			#print("Model 2: ", mdl2)
			#print("Model 3: ", mdl3)
			
			
			accuracies.append([j[0] for j in mdl1])
			accuracies.append([j[0] for j in mdl2])
			accuracies.append([j[0] for j in mdl3])
			precisions.append([j[1] for j in mdl1])
			precisions.append([j[1] for j in mdl2])
			precisions.append([j[1] for j in mdl3])
			recalls.append([j[2] for j in mdl1])
			recalls.append([j[2] for j in mdl2])
			recalls.append([j[2] for j in mdl3])
			

			print("Accuracies: ", accuracies)
			print("Precisions: ", precisions)
			print("Recalls: ", recalls)
			
			
			############################################# PLOTTING #################################################################
						
			# for batch size 500
			#accs = [[0.978, 0.98, 0.992, 0.992, 0.988, 0.984], [1.0, 1.0, 1.0, 1.0, 0.996, 1.0], [0.98, 0.98, 0.994, 0.992, 0.986, 0.982]]
			fully_accs = [list(zip(*[(ix+1,y) for ix,y in enumerate(x)])) for x in accuracies]
			for l in fully_accs:
				plt.plot(*l)
			plt.ylabel("accuracies")
			plt.xlabel("no_of_batches")
			plt.legend(["MultinomialNB", "SGD classifier", "BernoulliNB"], fontsize = 8, loc = 'lower right')
			plt.show()


			# for batch size 500
			#precs = [[0.9616724738675958, 0.9590163934426229, 0.984, 0.9857142857142858, 0.9784172661870504, 0.9709090909090909], [1.0, 1.0, 1.0, 1.0, 0.9927007299270073, 1.0], [0.965034965034965, 0.9590163934426229, 0.9879518072289156, 0.9857142857142858, 0.974910394265233, 0.967391304347826]]
			fully_precs = [list(zip(*[(ix+1,y) for ix,y in enumerate(x)])) for x in precisions]
			for l in fully_precs:
				plt.plot(*l)
			plt.ylabel("precisions")
			plt.xlabel("no_of_batches")
			plt.legend(["MultinomialNB", "SGD classifier", "BernoulliNB"], fontsize = 8, loc = 'lower right')
			plt.show()


			# for batch size 500
			#rec = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
			fully_rec = [list(zip(*[(ix+1,y) for ix,y in enumerate(x)])) for x in recalls]
			for l in fully_rec:
				plt.plot(*l)
			plt.ylabel("recall")
			plt.xlabel("no_of_batches")
			plt.legend(["MultinomialNB", "SGD classifier", "BernoulliNB"], fontsize = 8, loc = 'lower right')
			plt.show()
				
			
			
			n = 1


			
			###################################################################################################################
			
			
			

			
			
			


lines = ssc.socketTextStream("localhost", 6100)
#lines = dstream (array of RDDs)

lines.foreachRDD(df_fun)

ssc.start()
ssc.awaitTermination()
ssc.stop()

