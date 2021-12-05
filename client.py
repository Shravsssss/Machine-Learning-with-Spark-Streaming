import socket
import json
from pyspark.context import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from sklearn.feature_extraction.text import CountVectorizer
from pyspark.sql.functions import lower, col, udf, regexp_replace
import re
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from sklearn.feature_extraction import text


# Initialize the spark context.
sc = SparkContext(appName="SSML")
ssc = StreamingContext(sc, 5)

spark = SparkSession(sc)
schema = StructType([StructField("feature0", StringType(), True), StructField("feature1", StringType(), True), StructField("feature2", StringType(), True)])


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
		df = spark.createDataFrame(json.loads(lines.collect()[0]).values(), schema)
		removed0 = udf(removeall, StringType())
		new_df = df.withColumn("feature1", removed0(df["feature1"]))
		
		removed1 = udf(removeall, StringType())
		new_df1 = new_df.withColumn("feature0", removed1(new_df["feature0"]))
		
		
		new_df1.show()

lines = ssc.socketTextStream("localhost", 6100)
#lines = dstream (array of RDDs)
lines.foreachRDD(df_fun)

ssc.start()
ssc.awaitTermination()
ssc.stop()

