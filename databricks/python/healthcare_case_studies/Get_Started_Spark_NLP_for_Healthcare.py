# Databricks notebook source
# MAGIC %md
# MAGIC ## Get Started with Spark NLP for Healthcare

# COMMAND ----------

# MAGIC %md
# MAGIC ## Getting the keys and installation
# MAGIC 
# MAGIC 1. In order to get trial keys for Spark NLP for Healthcare
# MAGIC , fill the form at https://www.johnsnowlabs.com/spark-nlp-try-free/ and you will get your keys to your email in a few minutes.
# MAGIC 
# MAGIC 2. On a new cluster or existing one
# MAGIC 
# MAGIC   - add the following to the `Advanced Options -> Spark` tab, in `Spark.Config` box:
# MAGIC 
# MAGIC     ```bash
# MAGIC     spark.local.dir /var
# MAGIC     spark.kryoserializer.buffer.max 1000M
# MAGIC     spark.serializer org.apache.spark.serializer.KryoSerializer
# MAGIC     ```
# MAGIC   - add the following to the `Advanced Options -> Spark` tab, in `Environment Variables` box:
# MAGIC 
# MAGIC     ```bash
# MAGIC     AWS_ACCESS_KEY_ID=xxx
# MAGIC     AWS_SECRET_ACCESS_KEY=yyy
# MAGIC     SPARK_NLP_LICENSE=zzz
# MAGIC     ```
# MAGIC 
# MAGIC 3. Download the followings with AWS CLI to your local computer
# MAGIC 
# MAGIC     `$ aws s3 cp --region us-east-2 s3://pypi.johnsnowlabs.com/$jsl_secret/spark-nlp-jsl-$jsl_version.jar spark-nlp-jsl-$jsl_version.jar`
# MAGIC 
# MAGIC     `$ aws s3 cp --region us-east-2 s3://pypi.johnsnowlabs.com/$jsl_secret/spark-nlp-jsl/spark_nlp_jsl-$jsl_version-py3-none-any.whl spark_nlp_jsl-$jsl_version-py3-none-any.whl` 
# MAGIC 
# MAGIC 4. In `Libraries` tab inside your cluster:
# MAGIC 
# MAGIC  - Install New -> PyPI -> `spark-nlp==$public_version` -> Install
# MAGIC  - Install New -> Maven -> Coordinates -> `com.johnsnowlabs.nlp:spark-nlp_2.12:$public_version` -> Install
# MAGIC 
# MAGIC  - add following jars for the Healthcare library that you downloaded above:
# MAGIC         - Install New -> Python Whl -> upload `spark_nlp_jsl-$jsl_version-py3-none-any.whl`
# MAGIC 
# MAGIC         - Install New -> Jar -> upload `spark-nlp-jsl-$jsl_version.jar`
# MAGIC 
# MAGIC 5. Now you can attach your notebook to the cluster and use Spark NLP!
# MAGIC 
# MAGIC For more information, see 
# MAGIC 
# MAGIC   https://nlp.johnsnowlabs.com/docs/en/install#databricks-support
# MAGIC 
# MAGIC   https://nlp.johnsnowlabs.com/docs/en/licensed_install#install-spark-nlp-for-healthcare-on-databricks
# MAGIC   
# MAGIC The follwing notebook is prepared and tested on **r2.2xlarge at 8.0 (includes Apache Spark 3.1.1, Scala 2.12)** on Databricks
# MAGIC 
# MAGIC In order to get more detailed examples, please check this repository : https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare/databricks_notebooks

# COMMAND ----------

# MAGIC %md
# MAGIC Let's import the libraries which we will use in the following cells.

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
# MAGIC **Read Dataset**
# MAGIC 
# MAGIC We will download a sample file and create a spark dataframe.

# COMMAND ----------

! wget -q https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/pubmed/pubmed_sample_text_small.csv

# COMMAND ----------

pubMedDF = spark.read.option("header", "true").csv("dbfs:/pubmed_sample_text_small.csv")
                
pubMedDF.show(truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Clinical NER Pipeline
# MAGIC We will extract clinical entities from text by using `ner_clinical_large` model.

# COMMAND ----------

# Annotator that transforms a text column from dataframe into an Annotation ready for NLP
documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

# Sentence Detector annotator, processes various sentences per line

#sentenceDetector = SentenceDetector()\
        #.setInputCols(["document"])\
        #.setOutputCol("sentence")
sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")
 
# Tokenizer splits words in a relevant format for NLP
tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

# Clinical word embeddings trained on PubMED dataset
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")\
        .setInputCols(["sentence","token"])\
        .setOutputCol("embeddings")

pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
        .setInputCols(["sentence", "token"])\
        .setOutputCol("pos_tags")

# NER model trained on i2b2 (sampled from MIMIC) dataset
clinical_ner = MedicalNerModel.pretrained("ner_clinical_large","en","clinical/models")\
        .setInputCols(["sentence","token","embeddings"])\
        .setOutputCol("ner")

ner_converter = NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nerPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        pos_tagger,
        clinical_ner,
        ner_converter])


empty_data = spark.createDataFrame([[""]]).toDF("text")

ner_model = nerPipeline.fit(empty_data)

# COMMAND ----------

ner_model.stages

# COMMAND ----------

clinical_ner.getClasses()

# COMMAND ----------

result = ner_model.transform(pubMedDF.limit(100))

# COMMAND ----------

result.show()

# COMMAND ----------

result.select("sentence.result").show(truncate=100)

# COMMAND ----------

from pyspark.sql import functions as F

result_df = result.select(F.explode(F.arrays_zip("token.result","ner.result")).alias("cols"))\
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("ner_label"))

result_df.show(50, truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC Lets count the ner_labels.

# COMMAND ----------

result_df.select("token", "ner_label").groupBy('ner_label').count().orderBy('count', ascending=False).show(truncate=False)

# COMMAND ----------

result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.begin', 'ner_chunk.end', 'ner_chunk.metadata')).alias("cols")) \
      .select(F.expr("cols['3']['sentence']").alias("sentence_id"),
              F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']").alias("begin"),
              F.expr("cols['2']").alias("end"),
              F.expr("cols['3']['entity']").alias("ner_label")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **We can also filter NER results to get specific entities by using `setWhiteList()` parameter. In this example we will get only `PROBLEM` entities.**

# COMMAND ----------

ner_converter_filter = NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")\
        .setWhiteList(["PROBLEM"])

nerFilteredPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        clinical_ner,
        ner_converter_filter])


empty_data = spark.createDataFrame([[""]]).toDF("text")

ner_filtered_model = nerFilteredPipeline.fit(empty_data)

# COMMAND ----------

filtered_result = ner_filtered_model.transform(pubMedDF.limit(100))

# COMMAND ----------

filtered_result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.begin', 'ner_chunk.end', 'ner_chunk.metadata')).alias("cols")) \
               .select(F.expr("cols['3']['sentence']").alias("sentence_id"),
                       F.expr("cols['0']").alias("chunk"),
                       F.expr("cols['1']").alias("begin"),
                       F.expr("cols['2']").alias("end"),
                       F.expr("cols['3']['entity']").alias("ner_label")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **As you can see, we got only `PROBLEM` entities from the text.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### NER Visualization
# MAGIC 
# MAGIC We have sparknlp_display library for visualization. This library works with LightPipeline results.

# COMMAND ----------

sample_text = [pubMedDF.limit(3).collect()[i][0] for i in range(3)]

# COMMAND ----------

sample_text[1]

# COMMAND ----------

ner_lp = LightPipeline(ner_model)
light_result = ner_lp.fullAnnotate(sample_text[1])

# COMMAND ----------

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

vis = visualiser.display(light_result[0], label_col='ner_chunk', document_col='document', return_html=True)

# Change color of an entity label
#visualiser.set_label_colors({'PROBLEM':'#008080', 'TEST':'#800080', 'TREATMENT':'#808080'})
#visualiser.display(light_result[0], label_col='ner_chunk')

# Set label filter
# vis = visualiser.display(light_result, label_col='ner_chunk', document_col='document',
                   #labels=['PROBLEM','TEST','TREATMENT])
  
displayHTML(vis)

# COMMAND ----------

# MAGIC %md
# MAGIC **There are many NER models for different purposes in Spark NLP. Lets show what if we use `jsl_ner_wip_clinical` model that has about 80 different NER label.**

# COMMAND ----------

jsl_ner = MedicalNerModel.pretrained("jsl_ner_wip_clinical","en","clinical/models")\
        .setInputCols(["sentence","token","embeddings"])\
        .setOutputCol("ner")

jslPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        pos_tagger,
        jsl_ner,
        ner_converter])


empty_data = spark.createDataFrame([[""]]).toDF("text")

jsl_model = jslPipeline.fit(empty_data)

# COMMAND ----------

jsl_lp = LightPipeline(jsl_model)
jsl_light_result = jsl_lp.fullAnnotate(sample_text[1])

# COMMAND ----------

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

vis = visualiser.display(jsl_light_result[0], label_col='ner_chunk', document_col='document', return_html=True)

displayHTML(vis)

# COMMAND ----------

# MAGIC %md
# MAGIC **If you want to go over more about NER, you can check this comprehensive notebook :**
# MAGIC 
# MAGIC https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/databricks_notebooks/1.Clinical_Named_Entity_Recognition_Model_v3.0.ipynb

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Clinical Assertion
# MAGIC 
# MAGIC Now we will check the assertion status of the clinical entities. We will use `ner_clinical_large` model for NER detection, and `assertion_dl` model for checking the assertion status of detected entities. While doing that, we will use the same pipeline that we created fot detecting NER.

# COMMAND ----------

# Assertion model trained on i2b2 (sampled from MIMIC) dataset
# coming from sparknlp_jsl.annotator !!
clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
assertionPipeline = Pipeline(stages=[
    nerPipeline,
    clinical_assertion
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

assertion_model = assertionPipeline.fit(empty_data)

# COMMAND ----------

# MAGIC %md
# MAGIC **This time we will use LightPipeline while implementing.**

# COMMAND ----------

sample_text[0]

# COMMAND ----------

assertion_light = LightPipeline(assertion_model)

# COMMAND ----------

# MAGIC %md
# MAGIC **We can use `annotate` method to get faster results for short sentences.**

# COMMAND ----------

assertion_anno_res = assertion_light.annotate(sample_text[0])

# COMMAND ----------

assertion_anno_res.keys()

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets create a pandas dataframe to see our results obviously.**

# COMMAND ----------

pd.DataFrame(list(zip(assertion_anno_res["ner_chunk"], assertion_anno_res["assertion"])), columns=["ner_chunk", "assertion"])

# COMMAND ----------

# MAGIC %md
# MAGIC **This time we will use `fullAnnotate` method on our text to get metadata results.**

# COMMAND ----------

assertion_result = assertion_light.fullAnnotate(sample_text[0])[0]

# COMMAND ----------

assertion_result.keys()

# COMMAND ----------

chunks=[]
entities=[]
status=[]

for n,m in zip(assertion_result['ner_chunk'],assertion_result['assertion']):

    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    status.append(m.result)

df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status})

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md
# MAGIC **Also we can filter assertion results by using `AssertionFilterer` annotator. We will use the same pipeline that we vreated before to get the assertions. We will try to get only `present` assertions.**

# COMMAND ----------

assertion_filterer = AssertionFilterer()\
      .setInputCols("sentence","ner_chunk","assertion")\
      .setOutputCol("assertion_filtered")\
      .setWhiteList(["present"])

assertionFilteredPipeline = Pipeline(stages=[
    assertionPipeline,
    assertion_filterer
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

assertion_filtered_model = assertionFilteredPipeline.fit(empty_data)

# COMMAND ----------

assertion_filtered_light = LightPipeline(assertion_filtered_model)

# COMMAND ----------

assertion_filtered_result = assertion_filtered_light.fullAnnotate(sample_text[0])[0]

# COMMAND ----------

assertion_filtered_result.keys()

# COMMAND ----------

assertion_filtered_result["assertion_filtered"]

# COMMAND ----------

# MAGIC %md
# MAGIC Here is the `present` entities.

# COMMAND ----------

chunks=[]
entities=[]


for n in assertion_filtered_result['assertion_filtered']:

    chunks.append(n.result)
    entities.append(n.metadata['entity']) 


filtered_df = pd.DataFrame({'chunks':chunks, 'entities':entities})

filtered_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assertion Visualization
# MAGIC 
# MAGIC We can visualize the assertion status of detected entities by using `AssertionVisualizer` module of `sparknlp_display` library.

# COMMAND ----------

from sparknlp_display import AssertionVisualizer

assertion_vis = AssertionVisualizer()

## To set custom label colors:
assertion_vis.set_label_colors({'TREATMENT':'#008080', 'PROBLEM':'#800080'}) #set label colors by specifying hex codes

vis = assertion_vis.display(assertion_result, 
                            label_col = 'ner_chunk', 
                            assertion_col = 'assertion',
                            document_col = 'document' ,
                            return_html=True
                      )

displayHTML(vis)

# COMMAND ----------

# MAGIC %md
# MAGIC **If you want to go more over about assertion model examples, you can check this comprehensive notebook :**
# MAGIC 
# MAGIC https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Relation Extraction

# COMMAND ----------

# MAGIC %md
# MAGIC In this section, we will show an example of relation extraction models. We will use the same NER pipeline that we created before to extract clinical entities and `re_clinical` model to extract relations between these entities. The set of relations defined in the 2010 i2b2 relation challenge:
# MAGIC 
# MAGIC **TrIP:** A certain treatment has improved or cured a medical problem (eg, ‘infection resolved with antibiotic course’)
# MAGIC 
# MAGIC **TrWP:** A patient's medical problem has deteriorated or worsened because of or in spite of a treatment being administered (eg, ‘the tumor was growing despite the drain’)
# MAGIC 
# MAGIC **TrCP:** A treatment caused a medical problem (eg, ‘penicillin causes a rash’)
# MAGIC 
# MAGIC **TrAP:** A treatment administered for a medical problem (eg, ‘Dexamphetamine for narcolepsy’)
# MAGIC 
# MAGIC **TrNAP:** The administration of a treatment was avoided because of a medical problem (eg, ‘Ralafen which is contra-indicated because of ulcers’)
# MAGIC 
# MAGIC **TeRP:** A test has revealed some medical problem (eg, ‘an echocardiogram revealed a pericardial effusion’)
# MAGIC 
# MAGIC **TeCP:** A test was performed to investigate a medical problem (eg, ‘chest x-ray done to rule out pneumonia’)
# MAGIC 
# MAGIC **PIP:** Two problems are related to each other (eg, ‘Azotemia presumed secondary to sepsis’)

# COMMAND ----------

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos_tags", "token"])\
    .setOutputCol("dependencies")

clinical_re_Model = RelationExtractionModel()\
    .pretrained("re_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\
    #.setRelationPairs(["problem-test", "problem-treatment"]) we can set the possible relation pairs (if not set, all the relations will be calculated)

relPipeline = Pipeline(stages=[
    nerPipeline,
    dependency_parser,
    clinical_re_Model
])


empty_data = spark.createDataFrame([[""]]).toDF("text")

rel_model = relPipeline.fit(empty_data)

# COMMAND ----------

rel_model.stages

# COMMAND ----------

sample_text[1]

# COMMAND ----------

rel_light = LightPipeline(rel_model)
relation_res = rel_light.fullAnnotate(sample_text[1])[0]

# COMMAND ----------

relation_res.keys()

# COMMAND ----------

rel_pairs=[]
  
for rel in relation_res["relations"]:
    rel_pairs.append((
          rel.result, 
          rel.metadata['entity1'], 
          rel.metadata['entity1_begin'],
          rel.metadata['entity1_end'],
          rel.metadata['chunk1'], 
          rel.metadata['entity2'],
          rel.metadata['entity2_begin'],
          rel.metadata['entity2_end'],
          rel.metadata['chunk2'], 
          rel.metadata['confidence']
      ))

rel_df = pd.DataFrame(rel_pairs, columns=['relation','entity1','entity1_begin','entity1_end','chunk1','entity2','entity2_begin','entity2_end','chunk2', 'confidence'])
rel_df[rel_df.relation!="O"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Relation Visualization
# MAGIC 
# MAGIC We can visualize relations between entities by using `RelationExtractionVisualizer` module of `sparknlp_display` lìbrary.

# COMMAND ----------

from sparknlp_display import RelationExtractionVisualizer

re_vis = RelationExtractionVisualizer()

vis = re_vis.display(relation_res,
                     relation_col = 'relations',
                     document_col = 'document',
                     show_relations=True,
                     return_html=True)

displayHTML(vis)

# COMMAND ----------

# MAGIC %md
# MAGIC **If you want to go more over about relation extraction model examples, you can check this comprehensive notebook :**
# MAGIC 
# MAGIC https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Entity Resolution

# COMMAND ----------

# MAGIC %md
# MAGIC There are many entity resolution models for different kinds of purposes in Spark NLP. But mainly, we can collect these models in two categories:
# MAGIC 
# MAGIC * Chunk Entity Resolver Models
# MAGIC * Sentence Entity Resolver Models
# MAGIC 
# MAGIC Here are some of the resolver models in Spark NLP:
# MAGIC 
# MAGIC - sbiobertresolve_icd10cm 
# MAGIC - sbiobertresolve_icd10cm_augmented
# MAGIC - sbiobertresolve_icd10cm_slim_normalized
# MAGIC - sbiobertresolve_icd10cm_slim_billable_hcc
# MAGIC - sbertresolve_icd10cm_slim_billable_hcc_med
# MAGIC - sbiobertresolve_icd10pcs
# MAGIC - sbiobertresolve_snomed_findings (with clinical_findings concepts from CT version)
# MAGIC - sbiobertresolve_snomed_findings_int  (with clinical_findings concepts from INT version)
# MAGIC - sbiobertresolve_snomed_auxConcepts (with Morph Abnormality, Procedure, Substance, Physical Object, Body Structure concepts from CT version)
# MAGIC - sbiobertresolve_snomed_auxConcepts_int  (with Morph Abnormality, Procedure, Substance, Physical Object, Body Structure concepts from INT version)
# MAGIC - sbiobertresolve_rxnorm
# MAGIC - sbiobertresolve_rxcui
# MAGIC - sbiobertresolve_icdo
# MAGIC - sbiobertresolve_cpt
# MAGIC - sbiobertresolve_loinc
# MAGIC - sbiobertresolve_HPO
# MAGIC - sbiobertresolve_umls_major_concepts
# MAGIC - sbiobertresolve_umls_findings
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the same NER pipeline and `sbiobertresolve_icd10cm_slim_billable_hcc` ICD10 CM entity resolver model.

# COMMAND ----------

c2doc = Chunk2Doc()\
      .setInputCols("ner_chunk")\
      .setOutputCol("ner_chunk_doc") 

sbert_embedder = BertSentenceEmbeddings\
      .pretrained("sbert_jsl_medium_uncased",'en','clinical/models')\
      .setInputCols(["ner_chunk_doc"])\
      .setOutputCol("sbert_embeddings")

icd_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_icd10cm_slim_billable_hcc_med","en", "clinical/models") \
      .setInputCols(["ner_chunk", "sbert_embeddings"]) \
      .setOutputCol("icd10_code")\
      .setDistanceFunction("EUCLIDEAN")


resolverPipeline = Pipeline(stages=[
        nerPipeline,
        c2doc,
        sbert_embedder,
        icd_resolver
    
])

empty_data = spark.createDataFrame([[""]]).toDF("text")
resolver_model = resolverPipeline.fit(empty_data)

# COMMAND ----------

res_light = LightPipeline(resolver_model)

# COMMAND ----------

res_anno = res_light.annotate("bladder cancer")

# COMMAND ----------

res_anno

# COMMAND ----------

list(zip(res_anno["ner_chunk"], res_anno["icd10_code"]))

# COMMAND ----------

resolver_res = res_light.fullAnnotate(sample_text[1])[0]

# COMMAND ----------

resolver_res.keys()

# COMMAND ----------

chunks = []
codes = []
begin = []
end = []
resolutions= []
all_distances = []
all_codes= []
all_cosines = []
all_k_aux_labels= []
confidence = []
entity = []

for chunk, code in zip(resolver_res['ner_chunk'], resolver_res["icd10_code"]):

    begin.append(chunk.begin)
    entity.append(chunk.metadata['entity'])
    end.append(chunk.end)
    chunks.append(chunk.result)
    codes.append(code.result) 
    confidence.append(code.metadata['confidence'])
    all_codes.append(code.metadata['all_k_results'].split(':::'))
    resolutions.append(code.metadata['all_k_resolutions'].split(':::'))
    all_distances.append(code.metadata['all_k_distances'].split(':::'))
    all_cosines.append(code.metadata['all_k_cosine_distances'].split(':::'))
    all_k_aux_labels.append(code.metadata['all_k_aux_labels'].split(':::'))
    
df = pd.DataFrame({'chunks':chunks, 'entity':entity, 'begin': begin, 'end':end, 'code':codes, 'all_codes':all_codes, 
                   'resolutions':resolutions, 'all_k_aux_labels':all_k_aux_labels,'all_distances':all_cosines})



df['billable'] = df['all_k_aux_labels'].apply(lambda x: [i.split('||')[0] for i in x])
df['hcc_status'] = df['all_k_aux_labels'].apply(lambda x: [i.split('||')[1] for i in x])
df['hcc_score'] = df['all_k_aux_labels'].apply(lambda x: [i.split('||')[2] for i in x])
df['confidence'] = confidence

df = df.drop(['all_k_aux_labels'], axis=1)

# COMMAND ----------

pd.set_option("display.max_colwidth", 100)

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets check the confidence level > 0.5 results**

# COMMAND ----------

df[df.confidence.astype(float) > 0.5]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Entity Resolution Visualization

# COMMAND ----------

from sparknlp_display import EntityResolverVisualizer

er_vis = EntityResolverVisualizer()


## To set custom label colors:
er_vis.set_label_colors({'TREATMENT':'#800080', 'PROBLEM':'#77b5fe'}) #set label colors by specifying hex codes

vis = er_vis.display(resolver_res, 
                     label_col='ner_chunk', 
                     resolution_col = 'icd10_code',
                     document_col='document',
                     return_html=True)

displayHTML(vis)

# COMMAND ----------

# MAGIC %md
# MAGIC **If you want to go more over about entity resolution model examples, you can check this comprehensive notebooks :**
# MAGIC 
# MAGIC https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb
# MAGIC https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb

# COMMAND ----------

# MAGIC %md
# MAGIC ### End of Notebook
