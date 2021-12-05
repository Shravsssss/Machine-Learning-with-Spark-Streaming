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

# Initialize the spark context.
sc = SparkContext(appName="SSML")
ssc = StreamingContext(sc, 5)

spark = SparkSession(sc)
schema = StructType([StructField("feature0", StringType(), True), StructField("feature1", StringType(), True), StructField("feature2", StringType(), True)])

		
clf1 = MultinomialNB()
clf2 = SGDClassifier()
clf3 = BernoulliNB()

label_encoder = preprocessing.LabelEncoder()
vectorizer = HashingVectorizer(alternate_sign=False)


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

#lines: type rdd
def df_fun(lines):
	exists = len(lines.collect())
	if exists:
		
		################################################ PREPROCESSING ################################################
		
		df = spark.createDataFrame(json.loads(lines.collect()[0]).values(), schema)
		removed0 = udf(removeall, StringType())
		new_df = df.withColumn("feature1", removed0(df["feature1"]))
		
		removed1 = udf(removeall, StringType())
		new_df1 = new_df.withColumn("feature0", removed1(new_df["feature0"]))
		
		################################################################################################################################################
		"""
		tokenizer = Tokenizer(inputCol="feature0", outputCol="feature0_words")
		wordsData = tokenizer.transform(new_df1)
		
		
		hashingTF = HashingTF(inputCol="feature0_words", outputCol="feature0_rawFeatures")
		featurizedData = hashingTF.transform(wordsData)

		idf = IDF(inputCol="feature0_rawFeatures", outputCol="feature0_rawFeatures_features")
		idfModel = idf.fit(featurizedData)
		rescaledData = idfModel.transform(featurizedData)
		rescaledData = rescaledData.drop("feature0_rawFeatures", "feature0_words").withColumnRenamed("feature0_rawFeatures_features", "feature0_tfidf")"""

		############################################################
		
		"""
		tokenizer = Tokenizer(inputCol="feature1", outputCol="feature1_words")
		wordsData = tokenizer.transform(rescaledData)

		hashingTF = HashingTF(inputCol="feature1_words", outputCol="feature1_rawFeatures")
		featurizedData = hashingTF.transform(wordsData)

		idf = IDF(inputCol="feature1_rawFeatures", outputCol="feature1_rawFeatures_features")
		idfModel = idf.fit(featurizedData)
		rescaledData = idfModel.transform(featurizedData)
		rescaledData = rescaledData.drop("feature1_rawFeatures", "feature1_words").withColumnRenamed("feature1_rawFeatures_features", "feature1_tfidf")"""
		
			
		
		#rescaledData = rescaledData.withColumn("feature0_tfidf", toArray())
		#rescaledData = rescaledData.withColumn("feature1_tfidf", toArray())
		
		#####################################################################
		
		le = label_encoder.fit_transform(np.array([row["feature2"] for row in new_df1.collect()]))
		#le = np.array(le, dtype = 'numeric')
		

		#print(rescaledData.collect())
		re_data = new_df1.collect()
		#X = np.array([np.array(re_data[0][0]) , np.array(re_data[1][1])])
		X = vectorizer.fit_transform([" ".join([row["feature0"], row["feature1"]]) for row in re_data])
		#X = np.array([np.array([row["feature0_tfidf"], row["feature1_tfidf"]]) for row in re_data])

		
		#for row in dataCollect:
    	#	print(row['dept_name'] + "," +str(row['dept_id']))

		################################################ MODEL BUILDING ################################################

		
		X_train, X_test, y_train, y_test = train_test_split(X, le, test_size = 0.33)
		
		model1 = clf1.partial_fit(X_train, y_train, classes = np.unique(y_train))
		pred1 = model1.predict(X_test)
		
		ac1 = accuracy_score(y_test, pred1)
		prec1 = precision_score(y_test, pred1)
		rec1 = recall_score(y_test, pred1)
		conf_matrix1 = confusion_matrix(y_test, pred1)

		print("------------------Model 1-----------------")
		print("Accuracy Score: ", ac1)
		print("Precision Score: ", prec1)
		print("Recall Score: ", rec1)
		print("Confusion Matrix: \n", conf_matrix1)
		
		
		model2 = clf2.partial_fit(X_train, y_train, classes = np.unique(y_train))
		pred2 = model2.predict(X_test)
		
		ac2 = accuracy_score(y_test, pred2)
		prec2 = precision_score(y_test, pred2)
		rec2 = recall_score(y_test, pred2)
		conf_matrix2 = confusion_matrix(y_test, pred2)
		
		print("------------------Model 2-----------------")
		print("Accuracy Score: ", ac2)
		print("Precision Score: ", prec2)
		print("Recall Score: ", rec2)
		print("Confusion Matrix: \n", conf_matrix2)
		
		
		model3 = clf3.partial_fit(X_train, y_train, classes = np.unique(y_train))
		pred3 = model3.predict(X_test)
		
		ac3 = accuracy_score(y_test, pred3)
		prec3 = precision_score(y_test, pred3)
		rec3 = recall_score(y_test, pred3)
		conf_matrix3 = confusion_matrix(y_test, pred3)
		
		print("------------------Model 3-----------------")
		print("Accuracy Score: ", ac3)
		print("Precision Score: ", prec3)
		print("Recall Score: ", rec3)
		print("Confusion Matrix: \n", conf_matrix3)
		

		################################################################################################################################################





lines = ssc.socketTextStream("localhost", 6100)
#lines = dstream (array of RDDs)
lines.foreachRDD(df_fun)



ssc.start()
ssc.awaitTermination()
ssc.stop()

