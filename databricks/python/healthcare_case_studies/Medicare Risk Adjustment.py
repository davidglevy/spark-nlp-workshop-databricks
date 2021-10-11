# Databricks notebook source
# MAGIC %md
# MAGIC # Extracting Undiagnosed Conditions from Clinical Notes
# MAGIC 
# MAGIC ## Medicare Risk Adjustment: 
# MAGIC In the United States, the Centers for Medicare & Medicaid Services sets reimbursement for private Medicare plan sponsors based on the assessed risk of their beneficiaries. Information found in unstructured medical records may be more indicative of member risk than existing structured data, creating more accurate risk pools.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initial configurations

# COMMAND ----------

import os
import json
import string
import numpy as np
import pandas as pd

import sparknlp
import sparknlp_jsl
from sparknlp.base import *
from sparknlp.util import *
from sparknlp.annotator import *
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from sparknlp.pretrained import ResourceDownloader

from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from sparknlp.training import CoNLL

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 100)  
pd.set_option('display.expand_frame_repr', False)

print('sparknlp.version : ',sparknlp.version())
print('sparknlp_jsl.version : ',sparknlp_jsl.version())

spark


# COMMAND ----------

# MAGIC %md
# MAGIC ## Download oncology notes
# MAGIC 
# MAGIC In this notebook we will use the transcribed medical reports in [www.mtsamples.com](www.mtsamples.com). 
# MAGIC 
# MAGIC You can download those reports by the script [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/databricks/python/healthcare_case_studies/mt_scrapper.py).
# MAGIC     
# MAGIC We will use slightly modified version of some clinical notes which are downloaded from [www.mtsamples.com](www.mtsamples.com).

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create the folder which we will store the notes.

# COMMAND ----------

notes_path='/FileStore/HLS/nlp/data/'
delta_path='/FileStore/HLS/nlp/delta/jsl/'

dbutils.fs.mkdirs(notes_path)
os.environ['notes_path']=f'/dbfs{notes_path}'

# COMMAND ----------

# MAGIC %sh
# MAGIC cd $notes_path
# MAGIC wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/mt_oncology_10.zip
# MAGIC unzip mt_oncology_10.zip

# COMMAND ----------

dbutils.fs.ls(f'{notes_path}/mt_oncology_10')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Data and Write to Bronze Delta Layer
# MAGIC 
# MAGIC There are 50 clinical notes stored in delta table. We read the data and write the raw notes data into bronze delta tables

# COMMAND ----------

df = sc.wholeTextFiles(f'{notes_path}/mt_oncology_10/mt_note_0*.txt').toDF().withColumnRenamed('_1','path').withColumnRenamed('_2','text')
display(df.limit(5))

# COMMAND ----------

df.write.format('delta').mode('overwrite').save(f'{delta_path}/bronze/mt-oc-notes')
display(dbutils.fs.ls(f'{delta_path}/bronze/mt-oc-notes'))

# COMMAND ----------

sample_text = df.limit(3).select("text").collect()[0][0]
print(sample_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ICD-10 code extraction
# MAGIC Now, we will create a pipeline to extract ICD10 codes. This pipeline will find diseases and problems and then map their ICD10 codes. We will also check if this problem is still present or not.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")
 
sentenceDetector = SentenceDetectorDLModel.pretrained()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")
 
tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")\
 
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")
 
c2doc = Chunk2Doc()\
      .setInputCols("ner_chunk")\
      .setOutputCol("ner_chunk_doc") 
 
clinical_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")
 
ner_converter = NerConverter() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ner_chunk")\
      .setWhiteList(["Oncological", "Disease_Syndrome_Disorder", "Heart_Disease"])
 
sbert_embedder = BertSentenceEmbeddings\
      .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
      .setInputCols(["ner_chunk_doc"])\
      .setOutputCol("sbert_embeddings")
 
icd10_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc","en", "clinical/models")\
    .setInputCols(["ner_chunk", "sbert_embeddings"])\
    .setOutputCol("icd10cm_code")\
    .setDistanceFunction("EUCLIDEAN")\
    .setReturnCosineDistances(True)
 
clinical_assertion = AssertionDLModel.pretrained("jsl_assertion_wip", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
 
resolver_pipeline = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        clinical_ner,
        ner_converter,
        c2doc,
        sbert_embedder,
        icd10_resolver,
        clinical_assertion
    ])
 
data_ner = spark.createDataFrame([[""]]).toDF("text")
 
icd_model = resolver_pipeline.fit(data_ner)

# COMMAND ----------

# MAGIC %md
# MAGIC We can transform the data. In path column, we have long path. Instead we will use filename column. Every file name refers to different patient.

# COMMAND ----------

path_array = F.split(df['path'], '/')
df = df.withColumn('filename', path_array.getItem(F.size(path_array)- 1)).select(['filename', 'text'])

icd10_sdf = icd_model.transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see how our model extracted ICD Codes on a sample.

# COMMAND ----------

light_model = LightPipeline(icd_model)

light_result = light_model.fullAnnotate(sample_text)

from sparknlp_display import EntityResolverVisualizer

vis = EntityResolverVisualizer()

# Change color of an entity label
vis.set_label_colors({'PROBLEM':'#008080'})

icd_vis = vis.display(light_result[0], 'ner_chunk', 'icd10cm_code', return_html=True)

displayHTML(icd_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC ICD resolver can also tell us HCC status. HCC status is 1 if the Medicare Risk Adjusment model contains ICD code.

# COMMAND ----------

icd10_hcc_df = icd10_sdf.select("filename", F.explode(F.arrays_zip('ner_chunk.result', 
                                                                   'icd10cm_code.result',
                                                                   'icd10cm_code.metadata',
                                                                   "assertion.result"
                                                                  )).alias("cols")) \
                            .select("filename", F.expr("cols['0']").alias("chunk"),
                                    F.expr("cols['1']").alias("icd10_code"),
                                    F.expr("cols['2']['all_k_aux_labels']").alias("hcc_list"),
                                    F.expr("cols['3']").alias("assertion")
                                   ).toPandas()

# COMMAND ----------

icd10_hcc_df.head()

# COMMAND ----------

icd10_hcc_df["hcc_status"] = icd10_hcc_df["hcc_list"].apply(lambda x: x.split("||")[1])
icd10_df = icd10_hcc_df.drop("hcc_list", axis = 1)
icd10_df.head()

# COMMAND ----------

icd10_df = icd10_df[icd10_df.hcc_status=="1"]
icd10_df = icd10_df[~icd10_df.assertion.isin(["Family", "Past"])][['filename','chunk','icd10_code']].drop_duplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC We filtered the ICD codes based on HCC status. Now, we will create an ICD_code list column

# COMMAND ----------

icd10_df['Extracted_Entities_vs_ICD_Codes'] = list(zip(icd10_df.chunk, icd10_df.icd10_code))
icd10_df.head(10)

# COMMAND ----------

icd10_codes= icd10_df.groupby("filename").icd10_code.apply(lambda x: list(x)).reset_index()
icd10_vs_entities = icd10_df.groupby("filename").Extracted_Entities_vs_ICD_Codes.apply(lambda x: list(x)).reset_index()

icd10_df_all = icd10_codes.merge(icd10_vs_entities)

icd10_df_all

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gender Classification

# COMMAND ----------

# MAGIC %md
# MAGIC In Spark NLP, we have a pretrained model to detect gender of patient. Let's use it by `ClassifierDLModel`

# COMMAND ----------

documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

tokenizer = Tokenizer()\
      .setInputCols(["document"])\
      .setOutputCol("token")\

biobert_embeddings = BertEmbeddings().pretrained('biobert_pubmed_base_cased') \
        .setInputCols(["document",'token'])\
        .setOutputCol("bert_embeddings")

sentence_embeddings = SentenceEmbeddings() \
     .setInputCols(["document", "bert_embeddings"]) \
     .setOutputCol("sentence_bert_embeddings") \
     .setPoolingStrategy("AVERAGE")

genderClassifier = ClassifierDLModel.pretrained('classifierdl_gender_biobert', 'en', 'clinical/models') \
       .setInputCols(["document", "sentence_bert_embeddings"]) \
       .setOutputCol("gender")

gender_pipeline = Pipeline(stages=[documentAssembler,
                                   #sentenceDetector,
                                   tokenizer, 
                                   biobert_embeddings, 
                                   sentence_embeddings, 
                                   genderClassifier])

# COMMAND ----------

data_ner = spark.createDataFrame([[""]]).toDF("text")

gender_model = gender_pipeline.fit(data_ner)

gender_df = gender_model.transform(df)

# COMMAND ----------

gender_pd_df = gender_df.select("filename", F.explode(F.arrays_zip('gender.result', 'gender.metadata')).alias("cols")) \
                       .select("filename",
                               F.expr("cols['0']").alias("Gender"),
                               F.expr("cols['1']['Female']").alias("Female"),
                               F.expr("cols['1']['Male']").alias("Male")).toPandas()

gender_pd_df['Gender'] = gender_pd_df.apply(lambda x : "F" if float(x['Female']) >= float(x['Male']) else "M", axis=1)

gender_pd_df = gender_pd_df[['filename', 'Gender']]

# COMMAND ----------

# MAGIC %md
# MAGIC All patients' gender is ready in a dataframe.

# COMMAND ----------

gender_pd_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Age

# COMMAND ----------

# MAGIC %md
# MAGIC We can get patient's age from the notes by another pipeline. We are creating an age pipeline to get `AGE` labelled entities. In a note, more than one age entity can be extracted. We will get the first age entity as patient's age.

# COMMAND ----------

date_ner_converter = NerConverter() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ner_chunk")\
      .setWhiteList(["Age"])

age_pipeline = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        clinical_ner,
        date_ner_converter
    ])

data_ner = spark.createDataFrame([[""]]).toDF("text")

age_model = age_pipeline.fit(data_ner)

# COMMAND ----------

light_model = LightPipeline(age_model)

light_result = light_model.fullAnnotate(sample_text)

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

ner_vis = visualiser.display(light_result[0], label_col='ner_chunk', document_col='document', return_html=True)

displayHTML(ner_vis)

# COMMAND ----------

age_result = age_model.transform(df)

age_df = age_result.select("filename",F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
                   .select("filename", 
                           F.expr("cols['0']").alias("Age"),
                           F.expr("cols['1']['entity']").alias("ner_label")).toPandas().groupby('filename').first().reset_index()

# COMMAND ----------

age_df.head()

# COMMAND ----------

age_df.Age = age_df.Age.replace(r"\D", "", regex = True).astype(int)
age_df.drop('ner_label', axis=1, inplace=True)
age_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculating Medicare Risk Adjusment Score

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we have all data which can be extracted from clinical notes. Now we can calculate Medicare Risk Adjusment Score.

# COMMAND ----------

patient_df = age_df.merge(icd10_df_all, on='filename', how = "left")\
                   .merge(gender_pd_df, on='filename', how = "left").dropna()

#patient_df.icd10_code = patient_df.icd10_code.fillna("[]")

# COMMAND ----------

# MAGIC %md
# MAGIC There are more parameters used to calculate risk adjusment score. But those parameters may not be extracted from clinical notes. Let's add them manually.

# COMMAND ----------

imported_details = pd.DataFrame({"filename" : ['mt_note_01.txt', 'mt_note_03.txt', 'mt_note_05.txt', 
                                               'mt_note_06.txt', 'mt_note_08.txt', 'mt_note_09.txt'],
                                 "eligibility" : ["CFA", "CND", "CPA", "CFA", "CND", "CPA"],
                                 "orec" : ["0", "1", "3", "0", "1", "3"],
                                 "medicaid":[True, False, True, True, False, True]
                      })

patient_df = patient_df.merge(imported_details)

# COMMAND ----------

patient_df.head()

# COMMAND ----------

patient_sdf = spark.createDataFrame(patient_df[['filename','Age', 'icd10_code', 'Gender', 'eligibility', 'orec', 'medicaid']])
patient_sdf.show()

# COMMAND ----------

from pyspark.sql.types import MapType, IntegerType, DoubleType, StringType, StructType, StructField, FloatType
import pyspark.sql.functions as f

schema = StructType([
            StructField('risk_score', FloatType()),
            StructField('hcc_lst', StringType()),
            StructField('parameters', StringType()),
            StructField('details', StringType())])

# COMMAND ----------

from sparknlp_jsl.functions import profile

# COMMAND ----------

df = patient_sdf.withColumn("hcc_profile", profile(patient_sdf.icd10_code, 
                                                   patient_sdf.Age, 
                                                   patient_sdf.Gender,
                                                   patient_sdf.eligibility,
                                                   patient_sdf.orec,
                                                   patient_sdf.medicaid
                                                  ))

# COMMAND ----------

df.show()

# COMMAND ----------

df = df.withColumn("hcc_profile", F.from_json(F.col("hcc_profile"), schema))
df= df.withColumn("risk_score", df.hcc_profile.getItem("risk_score"))\
      .withColumn("hcc_lst", df.hcc_profile.getItem("hcc_lst"))\
      .withColumn("parameters", df.hcc_profile.getItem("parameters"))\
      .withColumn("details", df.hcc_profile.getItem("details"))

# COMMAND ----------

df.select('filename','risk_score','icd10_code', 'Age', 'Gender', 'eligibility', 'orec', 'medicaid').show(truncate=False )

# COMMAND ----------

df.show(truncate=100, vertical=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we have risk score of each patient!

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
# MAGIC |BeautifulSoup|MIT License|https://www.crummy.com/software/BeautifulSoup/#Download|https://www.crummy.com/software/BeautifulSoup/bs4/download/|
# MAGIC |Requests|Apache License 2.0|https://github.com/psf/requests/blob/main/LICENSE|https://github.com/psf/requests|
# MAGIC |Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
# MAGIC |Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
# MAGIC |Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
# MAGIC 
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
