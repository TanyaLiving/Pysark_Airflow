import os
import warnings
import yaml
import nltk
# import pandas as pd
# from sklearn.model_selection import StratifiedShuffleSplit
from functions import pipe
# import spark
import pyspark.pandas as ps
from pyspark.sql import SparkSession

nltk.download("omw-1.4")

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")

warnings.filterwarnings("ignore")

config_path = os.path.join("./config/config.yaml")
config = yaml.safe_load(open(config_path))["preparation"]

# Create SparkSession
spark = SparkSession.builder.getOrCreate()

## Read data
data = spark.read.csv(config["data_path"], sep=",", header=0)
print((sparkDF.count(), len(sparkDF.columns)))

# Splitting
zeros = data.filter(data["sentiment"]==0)
ones = data.filter(data["sentiment"]==1)

train0, test0 = zeros.randomSplit([0.8,0.2], seed=1234)
train1, test1 = ones.randomSplit([0.8,0.2], seed=1234)

train = train0.union(train1)
test = test0.union(test1)

# Data prepocessing
train_pipe = pipe(train)
train_pipe.write.csv("/Pyspark_Airflow/data/train_pipe.csv", sep=";")

# test pipe
test_pipe = pipe(test)
test_pipe.write.csv("/Pyspark_Airflow/data/test_pipe.csv", sep=";")