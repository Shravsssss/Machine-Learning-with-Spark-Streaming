import socket
import json
from pyspark.context import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
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

# Initialize the spark context.
sc = SparkContext(appName="SSML")
ssc = StreamingContext(sc, 5)

spark = SparkSession(sc)
schema = StructType([StructField("feature0", StringType(), True), StructField("feature1", StringType(), True), StructField("feature2", StringType(), True)])

global n
n = 1
global i
i = 0


mdl4 = list()

accuracies = list()
precisions = list()
recalls = list()

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
	#print("For n_clusters = ", k)
	print(f"Silhouette coefficient : {silhouette_score(X,km.labels_):0.2f}")
	print(f"Inertia:{km.inertia_}")
				
	return km, km.labels_

#lines: type rdd
def df_fun(lines):
	exists = len(lines.collect())
	if exists:
		global n
		global i
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


		################################################ CLUSTERING ################################################

		
		#X_train, X_test, y_train, y_test = train_test_split(X, le, test_size = 0.33)
		
		X_train = X
		y_train = le
		X_test = X
		y_test = le
		
		print("#batch", n)
		


		#X_kmeans = vectorizer.fit_transform([" ".join([row["feature0"], row["feature1"]]) for row in re_data])
		
		#c1, c_labels = mbkmeans_clusters(X_kmeans, k = 2, mb = new_df1.count(), print_silhouette_values=True)
		#print(c_labels)
		
		
		if n <= 60:
			kmm, km_labels = mbkmeans_clusters(X_train, k = 2, mb = new_df1.count(), print_silhouette_values=True)
			print("Cluster Training Done")			
			
		else:
			if n>60 and n<=66:
				pred4 = kmModel.predict(X_test)
				
				ac4 = accuracy_score(y_test, pred4)
				prec4 = precision_score(y_test, pred4)
				rec4 = recall_score(y_test, pred4)
				conf_matrix4 = confusion_matrix(y_test, pred4)
				mdl4.append([ac4, prec4, rec4])

				print("------------------Clustering Model-----------------")
				print("Accuracy Score: ", ac4)
				print("Precision Score: ", prec4)
				print("Recall Score: ", rec4)
				print("Confusion Matrix: \n", conf_matrix4)
				
				
			
				
		n += 1
		i += 1
		if n>66:
			accuracies.append([j[0] for j in mdl4])
			precisions.append([j[1] for j in mdl4])
			recalls.append([j[2] for j in mdl4])
			
			temp = accuracies + precisions + recalls
			#print(temp)
			#[[[0.67, 0.734, 0.72, 0.716, 0.71, 0.71]], [[0.9051094890510949, 0.8976377952755905, 0.9568965517241379, 0.958904109589041, 0.9440559440559441, 0.9357142857142857]], [[0.4492753623188406, 0.48717948717948717, 0.45121951219512196, 0.5072463768115942, 0.4963235294117647, 0.49063670411985016]]]


			
			perf_metrics = [list(zip(*[(ix+1,y) for ix,y in enumerate(x)])) for x in temp]
			#print(perf_metrics)
			#[[(1,), ([0.67, 0.734, 0.72, 0.716, 0.71, 0.71],)], [(1,), ([0.9051094890510949, 0.8976377952755905, 0.9568965517241379, 0.958904109589041, 0.9440559440559441, 0.9357142857142857],)], [(1,), ([0.4492753623188406, 0.48717948717948717, 0.45121951219512196, 0.5072463768115942, 0.4963235294117647, 0.49063670411985016],)]]
			
			
			for l in perf_metrics:
				plt.plot(*l)
			#plt.ylabel("accuracy")
			plt.xlabel("no_of_batches")
			plt.legend(["Accuracy", "Precision", "Recall"], fontsize = 8, loc = 'lower right')
			plt.show()
			
			n = 1


			
			################################################################################################################################################


lines = ssc.socketTextStream("localhost", 6100)
#lines = dstream (array of RDDs)

lines.foreachRDD(df_fun)

ssc.start()
ssc.awaitTermination()
ssc.stop()

