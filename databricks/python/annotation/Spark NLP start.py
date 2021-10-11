# Databricks notebook source
import sparknlp

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

from pyspark.ml import Pipeline


# COMMAND ----------

df = spark.createDataFrame([("Yeah, I get that. is the",)], ["comment"])
document_assembler = DocumentAssembler() \
    .setInputCol("comment") \
    .setOutputCol("document")
    
sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") \
    .setUseAbbreviations(True)
    
tokenizer = Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")
stemmer = Stemmer() \
    .setInputCols(["token"]) \
    .setOutputCol("stem")
    
normalizer = Normalizer() \
    .setInputCols(["stem"]) \
    .setOutputCol("normalized")

finisher = Finisher() \
    .setInputCols(["normalized"]) \
    .setOutputCols(["ntokens"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(True)

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, stemmer, normalizer, finisher])

nlp_model = nlp_pipeline.fit(df)
processed = nlp_model.transform(df).persist()

processed.count()
processed.show()

