# Databricks notebook source
# MAGIC %md
# MAGIC # Detecting Adverse Drug Events From Conversational Texts
# MAGIC 
# MAGIC Adverse Drug Events (ADEs) are potentially very dangerous to patients and are top causes of morbidity and mortality. Many ADEs are hard to discover as they happen to certain groups of people in certain conditions and they may take a long time to expose. Healthcare providers conduct clinical trials to discover ADEs before selling the products but normally are limited in numbers. Thus, post-market drug safety monitoring is required to help discover ADEs after the drugs are sold on the market. 
# MAGIC 
# MAGIC Less than 5% of ADEs are reported via official channels and the vast majority is described in free-text channels: emails & phone calls to patient support centers, social media posts, sales conversations between clinicians and pharma sales reps, online patient forums, and so on. This requires pharmaceuticals and drug safety groups to monitor and analyze unstructured medical text from a variety of jargons, formats, channels, and languages - with needs for timeliness and scale that require automation. 
# MAGIC 
# MAGIC Here we show how to use Spark NLP's existing models to process conversational text and extract highly specialized ADE and DRUG information that can be used for various downstream use cases, including;
# MAGIC <break>
# MAGIC - Conversational Texts ADE Classification
# MAGIC - Detecting ADE and Drug Entities From Texts
# MAGIC - Analysis of Drug and ADE Entities
# MAGIC - Finding Drugs and ADEs Have Been Talked Most
# MAGIC - Detecting Most Common Drug-ADE Pairs
# MAGIC - Checking Assertion Status of ADEs
# MAGIC - Relations Between ADEs and Drugs

# COMMAND ----------

# MAGIC %md
# MAGIC **Initial Configurations**

# COMMAND ----------

import json
import os

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel,Pipeline
from pyspark.sql import functions as F
from pyspark.sql.types import *

from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import sparknlp_jsl
import sparknlp

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth",100)

print('sparknlp.version : ',sparknlp.version())
print('sparknlp_jsl.version : ',sparknlp_jsl.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Dataset
# MAGIC 
# MAGIC We will use a slightly modified version of some conversational ADE texts which are downloaded from https://sites.google.com/site/adecorpus/home/document.
# MAGIC 
# MAGIC Also you can find an article about this dataset here: https://www.sciencedirect.com/science/article/pii/S1532046412000615
# MAGIC 
# MAGIC **We will work with two main files in the dataset:**
# MAGIC 
# MAGIC - ADE-AE.rel : Conversations with ADE.
# MAGIC - ADE-NEG.txt : Conversations with no ADE.
# MAGIC 
# MAGIC Lets get started with downloading these files.

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/ADE_Corpus_V2/DRUG-AE.rel -P /dbfs/
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/ADE_Corpus_V2/ADE-NEG.txt -P /dbfs/

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will create dataframes named `neg_df` with ADE-Negative conversations and `pos_df` with ADE-Positive conversations.

# COMMAND ----------

neg_df = spark.read.csv("/ADE-NEG.txt").select("_c0")\
                                       .withColumn("hash", F.hash("_c0"))\
                                       .orderBy("hash")

neg_df = neg_df.withColumn('text', F.split(neg_df["_c0"], 'NEG').getItem(1))\
               .withColumn("is_ADE", F.lit(False))\
               .drop_duplicates(["text"])\
               .withColumn("id", F.monotonically_increasing_id()).select("id", "text", "is_ADE")
              
display(neg_df.limit(20))

# COMMAND ----------

neg_df.count()

# COMMAND ----------

pos_df = spark.read.csv("/DRUG-AE.rel", sep="|", header=None).select("_c1")\
                                                             .withColumn("hash", F.hash("_c1"))\
                                                             .withColumnRenamed("_c1","text")\
                                                             .withColumn("is_ADE", F.lit(True))\
                                                             .orderBy("hash")\

pos_df = pos_df.drop_duplicates(["text"]).withColumn("id", F.monotonically_increasing_id()).select("id", "text", "is_ADE")
display(pos_df.limit(20))

# COMMAND ----------

pos_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC We will store our dataframes in delta-table.

# COMMAND ----------

delta_path='/FileStore/HLS/nlp/delta/jsl/'

neg_df.write.format('delta').mode('overwrite').save(f'{delta_path}/ADE/neg_df')
display(dbutils.fs.ls(f'{delta_path}/ADE/neg_df'))

# COMMAND ----------

pos_df.write.format('delta').mode('overwrite').save(f'{delta_path}/ADE/pos_df')
display(dbutils.fs.ls(f'{delta_path}/ADE/pos_df'))

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Conversational ADE Classification

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Use Case: Text Classification According To Contains ADE or Not

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will try to predict if a text contains ADE or not by using `classifierdl_ade_conversational_biobert`. For this, we will create a new dataframe merging all ADE negative and ADE positive texts and shuffle that.

# COMMAND ----------

sdf = neg_df.union(pos_df).drop("is_ADE").orderBy(F.rand(seed=42)).withColumn("id", F.monotonically_increasing_id())

display(sdf.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC Store the dataframe in a delta table.

# COMMAND ----------

sdf = sdf.repartition(12)

# COMMAND ----------

sdf.write.format('delta').mode('overwrite').save(f'{delta_path}/ADE/sdf_shuffled')
display(dbutils.fs.ls(f'{delta_path}/ADE/sdf_shuffled'))

# COMMAND ----------

document_assembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

tokenizer = Tokenizer()\
        .setInputCols(['document'])\
        .setOutputCol('token')

embeddings = BertEmbeddings.pretrained('biobert_pubmed_base_cased')\
        .setInputCols(["document", 'token'])\
        .setOutputCol("embeddings")

sentence_embeddings = SentenceEmbeddings() \
        .setInputCols(["document", "embeddings"]) \
        .setOutputCol("sentence_embeddings") \
        .setPoolingStrategy("AVERAGE")

conv_classifier = ClassifierDLModel.pretrained('classifierdl_ade_conversational_biobert', 'en', 'clinical/models')\
        .setInputCols(['document', 'token', 'sentence_embeddings'])\
        .setOutputCol('conv_class')


clf_pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer, 
    embeddings, 
    sentence_embeddings, 
    conv_classifier])

empty_data = spark.createDataFrame([['']]).toDF("text")
clf_model = clf_pipeline.fit(empty_data)

# COMMAND ----------

result = clf_model.transform(sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC Lets get the `classifierdl_ade_conversational_biobert` model results in `conv_cl_result` column.

# COMMAND ----------

 res_df = result.select("id", "text", F.explode(F.arrays_zip("conv_class.result")).alias("cols"))\
                .select("id", "text", F.expr("cols['0']").alias("conv_cl_result"))

# COMMAND ----------

display(res_df.limit(20))

# COMMAND ----------

res_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC We will change the type of the labels in our results to `boolean`.

# COMMAND ----------

res_df = res_df.withColumn('conv_cl_result', F.col('conv_cl_result').cast('boolean'))
display(res_df.limit(20))

# COMMAND ----------

res_df.printSchema()

# COMMAND ----------

res_df.write.format('delta').mode('overwrite').save(f'{delta_path}/ADE/clf_res')
display(dbutils.fs.ls(f'{delta_path}/ADE/clf_res'))

# COMMAND ----------

# MAGIC %md
# MAGIC **Let's see the number of prediction on each class.**

# COMMAND ----------

display(
  res_df.groupBy("conv_cl_result")
       .count()
       .orderBy(F.desc('count'))
       )

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets check some of the example sentences that the model predicted the ADE is `True` and `False`.**

# COMMAND ----------

display(res_df.filter(res_df["id"].isin([1,3,5,6,8,9,10])))

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. ADE-DRUG NER Examination
# MAGIC 
# MAGIC We will work on `pos_df` dataframe from now.

# COMMAND ----------

df = spark.read.format('delta').load(f'{delta_path}/ADE/pos_df/').drop("is_ADE")
display(df.limit(20))

# COMMAND ----------

df.count()

# COMMAND ----------

df = df.repartition(12)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1. Use Case: Detecting ADE and Drug Entities From Texts
# MAGIC 
# MAGIC Now we will extract `ADE` and `DRUG` entities from the conversational texts by using a combination of `ner_ade_clinical` and `ner_posology` models.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")
  
ade_ner = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ade_ner")

ade_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ade_ner"]) \
    .setOutputCol("ade_ner_chunk")\

pos_ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("pos_ner")

pos_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "pos_ner"]) \
    .setOutputCol("pos_ner_chunk")\
    .setWhiteList(["DRUG"])

chunk_merger = ChunkMergeApproach()\
    .setInputCols("ade_ner_chunk","pos_ner_chunk")\
    .setOutputCol("ner_chunk")\


ner_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    ade_ner,
    ade_ner_converter,
    pos_ner,
    pos_ner_converter,
    chunk_merger
    ])


empty_data = spark.createDataFrame([[""]]).toDF("text")
ade_ner_model = ner_pipeline.fit(empty_data)

# COMMAND ----------

result = ade_ner_model.transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Show the  `ADE` and `DRUG` phrases detected in conversations.**

# COMMAND ----------

display(result.select('id', 'text','ner_chunk.result')\
              .toDF('id', 'text','ADE_phrases')\
              .filter(F.size('ADE_phrases')>0)\
              .orderBy("id").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC **Show extracted chunks and their confidence levels**

# COMMAND ----------

result_sdf = result.select('id', F.explode(F.arrays_zip("ner_chunk.result","ner_chunk.metadata")).alias("cols"))\
                  .select('id', F.expr("cols['0']").alias("chunk"),
                                F.expr("cols['1']['entity']").alias("entity"),
                                F.expr("cols['1']['confidence']").alias("confidence")).orderBy("id")

# COMMAND ----------

display(result_sdf.limit(20))

# COMMAND ----------

result_sdf.write.format('delta').mode('overwrite').save(f'{delta_path}/ADE/ner_result')
display(dbutils.fs.ls(f'{delta_path}/ADE/ner_result'))

# COMMAND ----------

# MAGIC %md
# MAGIC **Highlight the extracted entities on the raw text by using `sparknlp_display` library for better visual understanding.**

# COMMAND ----------

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

light_model = LightPipeline(ade_ner_model)

sample_text = result.filter(result["id"].isin([0,2,5,7,12,13])).select(["text"]).collect()

for index, text in enumerate(sample_text):

    print("\n", "*"*50, f'Sample Text {index+1}', "*"*50, "\n")

    light_result = light_model.fullAnnotate(text)

    # change color of an entity label
    visualiser.set_label_colors({'ADE':'#ff037d', 'DRUG':'#7EBF9B'})
    
    ner_vis = visualiser.display(light_result[0], label_col='ner_chunk', document_col='document', return_html=True)
    
    displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Use Case: Analyse DRUG & ADE Entities - Find the DRUGs and ADEs have been talked most

# COMMAND ----------

# MAGIC %md
# MAGIC **Let's start by reading our table as a pandas dataframe and create `ADE` and `DRUG` dataframes.**

# COMMAND ----------

result_df = pd.read_parquet(f'/dbfs{delta_path}/ADE/ner_result')
result_df.head(20)

# COMMAND ----------

drug_df = result_df[result_df.entity == "DRUG"]
drug_df

# COMMAND ----------

ade_df = result_df[result_df.entity == "ADE"]
ade_df

# COMMAND ----------

# MAGIC %md
# MAGIC **We convert the chunks of these dataframes to lowercase to get more accurate results and check most frequent `ADE` and `DRUG` entities.**

# COMMAND ----------

drug_df.chunk = drug_df.chunk.str.lower()
drug_df.chunk.value_counts().head(20)

# COMMAND ----------

ade_df.chunk = ade_df.chunk.str.lower()
ade_df.chunk.value_counts().head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets show the talked most common `DRUG` and `ADE` entities on a barplot.**

# COMMAND ----------

import plotly.express as px

data=drug_df.chunk.value_counts().head(30)
data_pdf=pd.DataFrame({"Count":data.values,'Drug':data.index})
fig = px.bar(data_pdf, y='Drug', x='Count',orientation='h',color='Count', 
             color_continuous_scale=px.colors.sequential.Bluered, width=1200, height=700) 

fig.update_layout(
    title={
        'text': "Most Common DRUG Entities",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center'
        },
  font=dict(size=15))

fig.show()

# COMMAND ----------

import plotly.express as px

data=ade_df.chunk.value_counts().head(30)
data_pdf=pd.DataFrame({"Count":data.values,'ADE':data.index})
fig = px.bar(data_pdf, y='ADE', x='Count',orientation='h',color='Count', 
             color_continuous_scale=px.colors.sequential.Bluered, width=1200, height=700) 

fig.update_layout(
    title={
        'text': "Most Common ADE Entities",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center'
        },
    font=dict(size=15))

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Get Assertion Status of ADE & DRUG Entities
# MAGIC We will create a new pipeline by setting a WhiteList in `NerConverter` to get only `ADE` entities which comes from `ner_ade_clinical` model. Also will add the `assertion_jsl` model to get the assertion status of them. We can use the same annotators that are common with the NER pipeline we created before.

# COMMAND ----------

ade_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ade_ner"]) \
    .setOutputCol("ade_ner_chunk")\
    .setWhiteList(["ADE"])
 
 
assertion = AssertionDLModel.pretrained("assertion_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "ade_ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
 
 
assertion_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    ade_ner,
    ade_ner_converter,
    assertion
])
 
empty_data = spark.createDataFrame([[""]]).toDF("text")
assertion_model = assertion_pipeline.fit(empty_data)

# COMMAND ----------

as_result = assertion_model.transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Now we will create a dataframe with `ADE` chunks and their assertion status and the confidence level of results.**

# COMMAND ----------

as_result_sdf = as_result.select('id', 'text', F.explode(F.arrays_zip("ade_ner_chunk.result","ade_ner_chunk.metadata", "assertion.result")).alias("cols"))\
                        .select('id', 'text', F.expr("cols['0']").alias("chunk"),
                                        F.expr("cols['1']['entity']").alias("entity"),
                                        F.expr("cols['2']").alias("assertion"),
                                        F.expr("cols['1']['confidence']").alias("confidence")).orderBy('id')                 

# COMMAND ----------

as_result_sdf.write.format('delta').mode('overwrite').save(f'{delta_path}/ADE/assertion_result')
display(dbutils.fs.ls(f'{delta_path}/ADE/assertion_result'))

# COMMAND ----------

# MAGIC %md
# MAGIC **Show the assertion status of the entities on the raw text.**

# COMMAND ----------

from sparknlp_display import AssertionVisualizer

assertion_vis = AssertionVisualizer()

as_light_model = LightPipeline(assertion_model)

sample_text = df.filter(df["id"].isin([ 0, 10, 12, 28, 839])).select(["text"]).collect()

for index, text in enumerate(sample_text):

    as_light_result = as_light_model.fullAnnotate(text)

    print("\n", "*"*50, f'Sample Text {index+1}', "*"*50, "\n")
    
    assertion_vis.set_label_colors({'ADE':'#113CB8'})

    assert_vis =     assertion_vis.display(as_light_result[0], 
                                            label_col = 'ade_ner_chunk', 
                                            assertion_col = 'assertion', 
                                            document_col = 'document',
                                            return_html=True
                                            )
    displayHTML(assert_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets plot the assertion status counts of the `ADE` entities.**

# COMMAND ----------

display(
  as_result_sdf
  .groupBy('assertion')
  .count()
  .orderBy(F.desc('count'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Use Case: Conversation Counts by DRUG & ADE Entities
# MAGIC **We will work with the ADE entities by droping the assertion status is `absent`.**

# COMMAND ----------

final_ade_df = as_result_sdf.filter("assertion != 'Absent'").toPandas()
final_ade_df.chunk = final_ade_df.chunk.str.lower()

final_ade_df.head(30)

# COMMAND ----------

# MAGIC %md
# MAGIC **We will find the most frequent `ADE` and `DRUG` entities, then plot them on a chart to show the count of distinct conversations that contains these entities.**

# COMMAND ----------

most_common_ade = final_ade_df.chunk.value_counts().index[:20]
most_common_ade

# COMMAND ----------

import plotly.express as px

unique_ade = final_ade_df[final_ade_df.chunk.isin(most_common_ade)].rename(columns={"chunk":"ade"}).groupby(['id','ade']).count().reset_index()[['id', 'ade']]

data=unique_ade.ade.value_counts().head(20)
data_pdf=pd.DataFrame({"Count":data.values,'ADE':data.index})
fig = px.bar(data_pdf, y='ADE', x='Count',orientation='h',color='Count', 
             color_continuous_scale=px.colors.sequential.Bluyl, width=1200, height=700) 

fig.update_layout(
    title={
        'text': "Unique Conversation Counts by ADE",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center'
        },
    font=dict(size=15))

fig.show()

# COMMAND ----------

most_common_drug = drug_df.chunk.value_counts().index[:20]
most_common_drug

# COMMAND ----------

import plotly.express as px

unique_drug = drug_df[drug_df.chunk.isin(most_common_drug)].rename(columns={"chunk":"drug"}).groupby(['id','drug']).count().reset_index()[['id', 'drug']]

data=unique_drug.drug.value_counts().head(20)
data_pdf=pd.DataFrame({"Count":data.values,'ADE':data.index})
fig = px.bar(data_pdf, y='ADE', x='Count',orientation='h',color='Count', 
             color_continuous_scale=px.colors.sequential.Bluyl, width=1200, height=700) 

fig.update_layout(
    title={
        'text': "Unique Conversation Counts by DRUG",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center'
        },
    font=dict(size=15))

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Use Case: Most Common ADE-DRUG Pairs 
# MAGIC **We can find the most common ADE-DRUG pairs that were talked in the same conversation.**

# COMMAND ----------

top_20_ade = unique_ade.groupby("ade").count().sort_values(by="id", ascending=False).iloc[:20].index
top_20_drug = unique_drug.groupby("drug").count().sort_values(by="id", ascending=False).iloc[:20].index

# COMMAND ----------

merged_df = pd.merge(unique_ade[unique_ade.ade.isin(top_20_ade)],
                     unique_drug[unique_drug.drug.isin(top_20_drug)],
                     on = "id").groupby(["ade", "drug"]).count().reset_index()

drug_ade_df = merged_df.pivot_table(index="ade", columns=["drug"], values="id", fill_value=0)
drug_ade_df

# COMMAND ----------

import plotly.express as px

fig = px.imshow(drug_ade_df,labels=dict(x="DRUG", y="ADE", color='Occurence'),y=list(drug_ade_df.index), 
                x=list(drug_ade_df.columns), color_continuous_scale=px.colors.sequential.Mint)

fig.update_layout(
    autosize=False,
    width=1000,
    height=1000,
    title={
        'text': "Number of Conversation",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center'
        },
    font=dict(size=15))

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **As you can see in the results, Pneumonitis-Methotrexate is the most common `ADE-DRUG` pair.**

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Analyze Relations Between ADE & DRUG Entities

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1. Use Case: Extract Relations Between ADE and DRUG Entities

# COMMAND ----------

# MAGIC %md
# MAGIC We can extract the relations between `ADE` and `DRUG` entities by using `re_ade_clinical` model. We won't use `SentenceDetector` annotator in this pipeline to check the relations between entities in difference sentences.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")
  
ade_ner = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")

pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos_tags")    

dependency_parser = DependencyParserModel.pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos_tags", "token"])\
    .setOutputCol("dependencies")

reModel = RelationExtractionModel.pretrained("re_ade_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(0)\
    .setRelationPairs(["ade-drug", "drug-ade"])


re_pipeline = Pipeline(stages=[
    documentAssembler, 
    tokenizer,
    word_embeddings,
    ade_ner,
    ner_converter,
    pos_tagger,
    dependency_parser,
    reModel
])


empty_data = spark.createDataFrame([[""]]).toDF("text")
re_model = re_pipeline.fit(empty_data)

# COMMAND ----------

re_result = re_model.transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Now we can show our detected entities, their relations and confidence levels in a dataframe.**

# COMMAND ----------

rel_sdf = re_result.select('text', F.explode(F.arrays_zip('relations.result', 'relations.metadata')).alias("cols"))\
                  .select('text', F.expr("cols['0']").alias("relation"),
                                  F.expr("cols['1']['entity1']").alias("entity1"),
                                  F.expr("cols['1']['chunk1']").alias("chunk1"),
                                  F.expr("cols['1']['entity2']").alias("entity2"),
                                  F.expr("cols['1']['chunk2']").alias("chunk2"),
                                  F.expr("cols['1']['confidence']").alias("confidence"))

# COMMAND ----------

rel_sdf.write.format('delta').mode('overwrite').save(f'{delta_path}/ADE/relation_result')
display(dbutils.fs.ls(f'{delta_path}/ADE/relation_result'))

# COMMAND ----------

rel_df = pd.read_parquet(f'/dbfs{delta_path}/ADE/relation_result/')
rel_df.head(30)

# COMMAND ----------

# MAGIC %md
# MAGIC **We will convert the chunks to lowercase to get more accurate results.**

# COMMAND ----------

rel_df.chunk1 = rel_df.chunk1.str.lower()
rel_df.chunk2 = rel_df.chunk2.str.lower()

# COMMAND ----------

rel_df.drop_duplicates(["chunk1", "chunk2"]).head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC **We can show only `ADE-DRUG` in relation pairs.**

# COMMAND ----------

in_relation_df = rel_df[rel_df.relation.astype(int) == 1].drop_duplicates().reset_index(drop=True)
in_relation_df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC **Show the relations on the raw text bu using `sparknlp_display` library.**

# COMMAND ----------

from sparknlp_display import RelationExtractionVisualizer

re_light_model = LightPipeline(re_model)
re_vis = RelationExtractionVisualizer()

sample_text = df.filter(df["id"].isin([12, 34, 29, 4256, 1649])).select(["text"]).collect()

for index, text in enumerate(sample_text):

    print("\n", "*"*50, f'Sample Text {index+1}', "*"*50, "\n")
    
    re_light_result = re_light_model.fullAnnotate(text)

    relation_vis = re_vis.display(re_light_result[0],
                                  relation_col = 'relations',
                                  document_col = 'sentence',
                                  show_relations=True,
                                  return_html=True
                                   )
    
    displayHTML(relation_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library License|Library License URL|Library Source URL|
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
# MAGIC |Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
# MAGIC |Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
# MAGIC |MatPlotLib | | https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE | https://github.com/matplotlib/matplotlib|
# MAGIC |Seaborn |BSD 3-Clause License | https://github.com/seaborn/seaborn/blob/master/LICENSE | https://github.com/seaborn/seaborn/|
# MAGIC |Plotly|MIT License|https://github.com/plotly/plotly.py/blob/master/LICENSE.txt|https://github.com/plotly/plotly.py|
# MAGIC |Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
# MAGIC |Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
# MAGIC |Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
# MAGIC 
# MAGIC 
# MAGIC |Author|
# MAGIC |-|
# MAGIC |Databricks Inc.|
# MAGIC |John Snow Labs Inc.|

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disclaimers
# MAGIC Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.
