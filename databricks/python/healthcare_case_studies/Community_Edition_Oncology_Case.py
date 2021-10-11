# Databricks notebook source
# MAGIC %md
# MAGIC ## Getting the keys and installation
# MAGIC 
# MAGIC 1. In order to get trial keys for Spark NLP for Healthcare
# MAGIC , fill the form at https://www.johnsnowlabs.com/spark-nlp-try-free/ and you will get your keys to your email in a few minutes.
# MAGIC 2. On a new cluster or existing one
# MAGIC   - add the following to the `Advanced Options -> Spark` tab, in `Spark.Config` box:
# MAGIC     ```bash
# MAGIC     spark.local.dir /var
# MAGIC     spark.kryoserializer.buffer.max 1000M
# MAGIC     spark.serializer org.apache.spark.serializer.KryoSerializer
# MAGIC     ```
# MAGIC   - add the following to the `Advanced Options -> Spark` tab, in `Environment Variables` box:
# MAGIC     ```bash
# MAGIC     AWS_ACCESS_KEY_ID=xxx
# MAGIC     AWS_SECRET_ACCESS_KEY=yyy
# MAGIC     SPARK_NLP_LICENSE=zzz
# MAGIC     ```
# MAGIC 3. Download the followings with AWS CLI to your local computer
# MAGIC 
# MAGIC     `$ aws s3 cp --region us-east-2 s3://pypi.johnsnowlabs.com/$jsl_secret/spark-nlp-jsl-$jsl_version.jar spark-nlp-jsl-$jsl_version.jar`
# MAGIC     `$ aws s3 cp --region us-east-2 s3://pypi.johnsnowlabs.com/$jsl_secret/spark-nlp-jsl/spark_nlp_jsl-$jsl_version-py3-none-any.whl spark_nlp_jsl-$jsl_version-py3-none-any.whl` 
# MAGIC     
# MAGIC 4. In `Libraries` tab inside your cluster:
# MAGIC 
# MAGIC  - Install New -> PyPI -> `spark-nlp==$public_version` -> Install
# MAGIC  - Install New -> Maven -> Coordinates -> `com.johnsnowlabs.nlp:spark-nlp_2.12:$public_version` -> Install
# MAGIC  - add following jars for the Healthcare library that you downloaded above:
# MAGIC         - Install New -> Python Whl -> upload `spark_nlp_jsl-$jsl_version-py3-none-any.whl`
# MAGIC         - Install New -> Jar -> upload `spark-nlp-jsl-$jsl_version.jar`
# MAGIC         
# MAGIC 5. Now you can attach your notebook to the cluster and use Spark NLP!
# MAGIC 
# MAGIC For more information, see 
# MAGIC   https://nlp.johnsnowlabs.com/docs/en/install#databricks-support
# MAGIC   https://nlp.johnsnowlabs.com/docs/en/licensed_install#install-spark-nlp-for-healthcare-on-databricks

# COMMAND ----------

# MAGIC %md
# MAGIC # Abstracting Real World Data from Oncology Notes
# MAGIC MT ONCOLOGY NOTES comprises of millions of ehr records of patients. It contains structured data like demographics, insurance details, and a lot more, but most importantly, it also contains free-text data like real encounters and notes.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Goal: We'll see how we can use Spark NLP's existing models to process raw text and extract highly specialized cancer information for various use cases.
# MAGIC 
# MAGIC - Staff demand analysis according to specialties.
# MAGIC - Preparing reimbursement-ready data with billable codes.
# MAGIC - Analysis of risk factors of patients and symptoms.
# MAGIC - Analysis of cancer disease and symptoms.
# MAGIC - Drug usage analysis for inventory management.
# MAGIC - Preparing timeline of procedures.
# MAGIC - Relations between internal body part and procedures.
# MAGIC - Analysis of procedures used on oncological events.
# MAGIC - Checking assertion status of oncological findings.

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
# MAGIC **!!! Warning :** This notebook is optimized to run in Databricks CE. If you want to test the same Spark NLP for Helathcare modules with large data, please consider upgrading to larger DBUs. Also this notebook can only work with Spark NLP for Healthcare `v3.1.0` an above. If you have previous versions, please consider upgrading your version.

# COMMAND ----------

# MAGIC %sh
# MAGIC for i in {0..2}
# MAGIC do
# MAGIC  wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/oncology_notes/mt_oncology_$i.txt
# MAGIC done

# COMMAND ----------

notes_path='file:/databricks/driver/mt_oncology_*.txt'



textFiles = spark.sparkContext.wholeTextFiles(notes_path)

df = textFiles.toDF(schema=['path','text'])

# COMMAND ----------

# MAGIC %md
# MAGIC **Vizualize the Entities Using Spark NLP Display Library**

# COMMAND ----------

# MAGIC %md
# MAGIC At first, we will create a NER pipeline. And then, we can see the labbeled entities on text.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")

tokenizer = Tokenizer()\
  .setInputCols(["sentence"])\
  .setOutputCol("token")\

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")

# Cancer
bionlp_ner = MedicalNerModel.pretrained("ner_bionlp", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("bionlp_ner")

bionlp_ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "bionlp_ner"]) \
  .setOutputCol("bionlp_ner_chunk")\
  .setWhiteList(["Cancer"])

# Clinical Terminology
jsl_ner = MedicalNerModel.pretrained("jsl_ner_wip_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("jsl_ner")

jsl_ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "jsl_ner"]) \
  .setOutputCol("jsl_ner_chunk")\
  .setWhiteList(["Oncological", "Symptom", "Treatment"])

# COMMAND ----------

# MAGIC %md
# MAGIC We used two diiferent NER model (`jsl_ner_wip_clinical` and `bionlp_ner`) and we need to merge the by a chunk merger. There are two different entities related to oncology. So we will change `Cancer` entities to `Oncological` by `setReplaceDictResource` parameter. This parameter gets the list from a csv file. Before merging the entities, we are creating the csv file with a row `Cancer,Oncological`.

# COMMAND ----------

replace_dict = 'Cancer,Oncological'
with open('replace_dict.csv', 'w') as f:
    f.write(replace_dict)
    
dbutils.fs.cp("file:/databricks/driver/replace_dict.csv", "dbfs:/", recurse=True)

# COMMAND ----------

chunk_merger = ChunkMergeApproach()\
  .setInputCols("bionlp_ner_chunk","jsl_ner_chunk")\
  .setOutputCol("final_ner_chunk")\
  .setReplaceDictResource("dbfs:/replace_dict.csv","text", {"delimiter":","})
 
ner_pipeline= Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        bionlp_ner,
        bionlp_ner_converter,
        jsl_ner,
        jsl_ner_converter,
        chunk_merger])
empty_data = spark.createDataFrame([['']]).toDF("text")
ner_model = ner_pipeline.fit(empty_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will visualize a sample text with `NerVisualizer`.
# MAGIC 
# MAGIC `NerVisualizer` woks with Lightpipeline, so we will create a `light_model` with our `icd10_model`.

# COMMAND ----------

sample_text = df.limit(1).select("text").collect()[0]

# COMMAND ----------

light_model =  LightPipeline(ner_model)
 
ann_text = light_model.fullAnnotate(sample_text)[0]
ann_text.keys()

# COMMAND ----------

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

# Change color of an entity label
visualiser.set_label_colors({'ONCOLOGICAL':'#ff2e51', 'TREATMENT': '#0A902E', 'SYMPTOM': '#7D087D' })

ner_vis = visualiser.display(ann_text, label_col='final_ner_chunk',return_html=True)

displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Get ICD-10 codes using entity resolvers and use the data for various use cases.
# MAGIC 
# MAGIC We can use hcc_billable entity resolver to get ICD10-CM codes for identified entities. The unique this about this resolver is it also provides HCC risk factor and billable status for each ICD code. We can use this information for a lot of tasks.

# COMMAND ----------

# MAGIC %md
# MAGIC We are inserting `sbiobert_base_cased_mli` embedding and `sbiobertresolve_icd10cm_augmented_billable_hcc` resolver.

# COMMAND ----------

chunk_merger = ChunkMergeApproach()\
  .setInputCols("bionlp_ner_chunk","jsl_ner_chunk")\
  .setOutputCol("final_ner_chunk")\
  .setReplaceDictResource("dbfs:/replace_dict.csv","text", {"delimiter":","})

c2doc = Chunk2Doc()\
  .setInputCols("final_ner_chunk")\
  .setOutputCol("final_ner_chunk_doc") 

sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", 'en', 'clinical/models')\
  .setInputCols(["final_ner_chunk_doc"])\
  .setOutputCol("sentence_embeddings")

icd10_resolver = SentenceEntityResolverModel.pretrained("demo_sbiobertresolve_icd10cm_augmented_billable_hcc","en", "clinical/models")\
  .setInputCols(["final_ner_chunk", "sentence_embeddings"]) \
  .setOutputCol("icd10_code")\
  .setDistanceFunction("EUCLIDEAN")

bert_pipeline_icd10cm = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        bionlp_ner,
        bionlp_ner_converter,
        jsl_ner,
        jsl_ner_converter,
        chunk_merger,
        c2doc, 
        sbert_embedder,
        icd10_resolver])
empty_data = spark.createDataFrame([['']]).toDF("text")
icd10_model = bert_pipeline_icd10cm.fit(empty_data)

# COMMAND ----------

# MAGIC %md
# MAGIC **Now we will transform our dataframe by using `icd10_model` and get the ICD10CM codes of the entities.**

# COMMAND ----------

icd10_hcc_res = icd10_model.transform(df)

# COMMAND ----------

icd10_hcc_df = icd10_hcc_res.select("path", F.explode(F.arrays_zip('final_ner_chunk.result', 
                                                                   'final_ner_chunk.metadata', 
                                                                   'icd10_code.result', 
                                                                   'icd10_code.metadata')).alias("cols")) \
                            .select("path", F.expr("cols['0']").alias("final_chunk"),
                                     F.expr("cols['1']['entity']").alias("entity"), 
                                     F.expr("cols['2']").alias("icd10_code"),
                                     F.expr("cols['3']['confidence']").alias("confidence"),
                                     F.expr("cols['3']['all_k_results']").alias("all_codes"),
                                     F.expr("cols['3']['all_k_resolutions']").alias("resolutions"),
                                     F.expr("cols['3']['all_k_aux_labels']").alias("hcc_list")).toPandas()

codes = []
resolutions = []
hcc_all = []

for code, resolution, hcc in zip(icd10_hcc_df['all_codes'], icd10_hcc_df['resolutions'], icd10_hcc_df['hcc_list']):
    
    codes.append( code.split(':::'))
    resolutions.append(resolution.split(':::'))
    hcc_all.append(hcc.split(":::"))

icd10_hcc_df['all_codes'] = codes  
icd10_hcc_df['resolutions'] = resolutions
icd10_hcc_df['hcc_list'] = hcc_all

# COMMAND ----------

# MAGIC %md
# MAGIC The values in `billable`, `hcc_store` and `hcc_status` columns are seperated by `||` and we will change them to a list.

# COMMAND ----------

def extract_billable(bil):
  
  billable = []
  status = []
  score = []

  for b in bil:
    billable.append(b.split("||")[0])
    status.append(b.split("||")[1])
    score.append(b.split("||")[2])

  return (billable, status, score)

icd10_hcc_df["hcc_status"] = icd10_hcc_df["hcc_list"].apply(extract_billable).apply(pd.Series).iloc[:,1]
icd10_hcc_df["hcc_score"] = icd10_hcc_df["hcc_list"].apply(extract_billable).apply(pd.Series).iloc[:,2]
icd10_hcc_df["billable"] = icd10_hcc_df["hcc_list"].apply(extract_billable).apply(pd.Series).iloc[:,0]

icd10_hcc_df.drop("hcc_list", axis=1, inplace= True)
icd10_hcc_df['icd_codes_names'] = icd10_hcc_df['resolutions'].apply(lambda x : x[0])
icd10_hcc_df['icd_code_billable'] = icd10_hcc_df['billable'].apply(lambda x : x[0])

# COMMAND ----------

icd10_hcc_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC We have three different entities in the dataframe.

# COMMAND ----------

icd10_hcc_df["entity"].value_counts()

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.xticks(rotation=90)
plt.title("Count of entities", size=20)
sns.countplot(icd10_hcc_df.entity, order=icd10_hcc_df["entity"].value_counts().index)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1. Get general information for staff management, reporting, & planning.
# MAGIC 
# MAGIC Let's map diagnosis codes to parent categories to see which indications have most cases. 
# MAGIC 
# MAGIC We have defined a function and applied it to `icd10_code`.

# COMMAND ----------

icd10_oncology_mapping = {"C81-C96": "Malignant neoplasms of lymphoid, hematopoietic and related tissue",
                          "C76-C80": "Malignant neoplasms of ill-defined, other secondary and unspecified sites",
                          "D00-D09": "In situ neoplasms",
                          "C51-C58": "Malignant neoplasms of female genital organs",
                          "C43-C44": "Melanoma and other malignant neoplasms of skin",
                          "C15-C26": "Malignant neoplasms of digestive organs",
                          "C73-C75": "Malignant neoplasms of thyroid and other endocrine glands",
                          "D60-D64": "Aplastic and other anemias and other bone marrow failure syndromes",
                          "E70-E88": "Metabolic disorders",
                          "G89-G99": "Other disorders of the nervous system",
                          "R50-R69": "General symptoms and signs",
                          "R10-R19": "Symptoms and signs involving the digestive system and abdomen",
                          "Z00-Z13": "Persons encountering health services for examinations"}


def map_to_parent(x):
    charcode = x[0].lower()
    numcodes = int(x[1])
    
    for k, v in icd10_oncology_mapping.items():
        
        lower, upper = k.split('-')
        
        if charcode >= lower[0].lower() and numcodes >= int(lower[1]):
            
            if charcode < upper[0].lower():
                return v
            elif charcode == upper[0].lower() and numcodes <= int(upper[1]):
                return v

# COMMAND ----------

icd10_hcc_df["onc_code_desc"] = icd10_hcc_df["icd10_code"].apply(map_to_parent).fillna("-")
icd10_hcc_df.onc_code_desc.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets plot a countplot to see the number of each parent categories.**

# COMMAND ----------

plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
temp_df = icd10_hcc_df[icd10_hcc_df.onc_code_desc != "-"]

sns.countplot(temp_df.onc_code_desc, order=temp_df.onc_code_desc.value_counts().index)
plt.title("Mapped ICDs", size=20)
plt.xlabel("Parent Categories")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2. Preparing reimbursement-ready data with billable codes
# MAGIC 
# MAGIC Here, we will check how many of the ICD codes are billable.

# COMMAND ----------

print(icd10_hcc_df['icd_code_billable'].value_counts())

# COMMAND ----------

plt.figure(figsize=(3,4), dpi=200)
plt.pie(icd10_hcc_df['icd_code_billable'].value_counts(), 
        labels = ["billable", "not billable"], 
        autopct = "%1.1f%%"
       )
plt.title("Ratio Billable & Non-billable Codes", size=10)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **As we can see, some of the best matching codes are not billable. For such indications we can find codes that are relevant as well as billable.**

# COMMAND ----------

best_paid_icd_matches = []
indication_with_no_billable_icd = []

for i_, row in icd10_hcc_df.iterrows():
    if '1' not in row['billable']:
        indication_with_no_billable_icd.append([row['final_chunk'], 
                                      row['resolutions'][0], 
                                      row['all_codes'][0],
                                      row['billable'][0],
                                      row['hcc_score'][0],
                                      row['onc_code_desc'], 
                                      "-" ])
    else:
        n_zero_ind = row['billable'].index('1')
        best_paid_icd_matches.append([row['final_chunk'], 
                                      row['resolutions'][n_zero_ind], 
                                      row['all_codes'][n_zero_ind],
                                      row['billable'][n_zero_ind],
                                      row['hcc_score'][n_zero_ind],
                                      row['onc_code_desc'],
                                      n_zero_ind])

best_icd_mapped = pd.DataFrame(best_paid_icd_matches, columns=['ner_chunk', 'code', 'code_desc', 'billable', 
                                             'corresponding_hcc_score', 'onc_code_desc', 'nearest_billable_code_pos'])
best_icd_mapped['corresponding_hcc_score'] = pd.to_numeric(best_icd_mapped['corresponding_hcc_score'], errors='coerce')

best_icd_mapped.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **All chunks have been mapped to payable ICD codes**

# COMMAND ----------

print(best_icd_mapped.billable.value_counts())

# COMMAND ----------

print("Number of non-billable ICD Codes: ",len(indication_with_no_billable_icd))

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets take a look at the position of nearest codes that was taken for making billable**

# COMMAND ----------

plt.figure(figsize=(6,4), dpi=150)
sns.countplot(best_icd_mapped['nearest_billable_code_pos'], order = best_icd_mapped.nearest_billable_code_pos.value_counts().index)
plt.title("Number of Nearest Billable Codes by Index", size=10)
plt.xlabel("Index of Nearest Billable Codes", size=10)
plt.text(4, 10, best_icd_mapped.nearest_billable_code_pos.value_counts().to_string())

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3. See which indications have highest average risk factor
# MAGIC 
# MAGIC In our pipeline we used `demo_sbiobertresolve_icd10cm_augmented_billable_hcc` as Sentence resolver. So the model return HCC codes. We can calculate the risk per indication by getting the averages.

# COMMAND ----------

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.figure(figsize=(25,6), dpi=150)

sns.barplot(x='onc_code_desc', y='corresponding_hcc_score',
            data = best_icd_mapped[best_icd_mapped.onc_code_desc != "-"], ci = None,
            palette="Set2")

plt.title('Average risk per indication', size=20)
plt.xlabel('diagnosis_categroy')
plt.ylabel('average_hcc_risk_code')
plt.xticks(rotation=90)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC **As we can see, some categories, even with fewer cases, have higher risk factor.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4. Analyze Oncological Entities
# MAGIC We can find the most frequent oncological entities.

# COMMAND ----------

onc_df = icd10_hcc_df[icd10_hcc_df.entity=="Oncological"].iloc[:, [0,1,2,3,10,11]]
onc_df.head()

# COMMAND ----------

print(onc_df.icd_codes_names.value_counts().head(20))

# COMMAND ----------

plt.figure(figsize=(20,6), dpi=150)
plt.xticks(rotation=90)
plt.title('Most Common Oncological Entities', size=20)
onc_df.icd_codes_names.value_counts().head(30).plot(kind='bar', color=['darkred', 'darkgreen'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5. Report Counts by ICD10CM Code Names
# MAGIC Each bar shows count of reports contain the cancer entities.

# COMMAND ----------

most_common_icd_codes = onc_df.icd_codes_names.value_counts().index[:20]
print(most_common_icd_codes)

# COMMAND ----------

unique_icd_code_names = onc_df[onc_df.icd_codes_names.isin(most_common_icd_codes)].groupby(["path","icd_codes_names"]).count().reset_index()[["path","icd_codes_names"]]	

plt.figure(figsize=(20,5), dpi=150)
plt.xticks(rotation=90)
plt.title('Unique Report Counts by ICD10CM')

sns.countplot(unique_icd_code_names.icd_codes_names, order=most_common_icd_codes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.6. Most common symptoms
# MAGIC  We can find the most common symptoms counting the unique symptoms in documents.

# COMMAND ----------

symptom_df = icd10_hcc_df[icd10_hcc_df.entity.isin(["Symptom"])].iloc[:, [0,1,2,3,10,11]]
symptom_df.rename(columns={"icd_codes_names":"symptom"}, inplace=True)
symptom_df.head()

# COMMAND ----------

unique_symptoms = symptom_df.groupby(["path","symptom"]).count().reset_index()[["path","symptom"]]	
most_common_symptoms = unique_symptoms.symptom.value_counts().index[:30]
print(most_common_symptoms)

# COMMAND ----------

plt.figure(figsize=(20,8), dpi=150)
plt.xticks(rotation=90)
plt.title('Most common symptoms', size=20)
sns.countplot(unique_symptoms.symptom, order=most_common_symptoms, palette="YlGnBu")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.7. Extract most frequent oncological diseases and symptoms based on documents
# MAGIC Here, we will count the number documents for each symptom-disease pair.

# COMMAND ----------

#Getting the list of the most common code names.
top_20_code_names = unique_icd_code_names.groupby("icd_codes_names").count().sort_values(by="path", ascending=False).iloc[:20].index
 
#Getting the list of common symptoms.
top_20_symptom = unique_symptoms.groupby("symptom").count().sort_values(by="path", ascending=False).iloc[:20].index

# COMMAND ----------

merged_df = pd.merge(unique_icd_code_names[unique_icd_code_names.icd_codes_names.isin(top_20_code_names)],
                     unique_symptoms[unique_symptoms.symptom.isin(top_20_symptom)],
                     on = "path").groupby(["icd_codes_names", "symptom"]).count().reset_index()
 
sympytom_cancer = merged_df.pivot_table(index="symptom", columns=["icd_codes_names"], values="path", fill_value=0)
 
sympytom_cancer

# COMMAND ----------

plt.figure(figsize=(8,6), dpi=120)
sns.heatmap(sympytom_cancer, annot=True, cbar=False, annot_kws={"size": 8}, cmap = "YlGnBu")
plt.title('Number of Patients')
plt.xlabel('Disease')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Get Drug codes from the notes
# MAGIC 
# MAGIC We will create a new pipeline to get drug codes. As NER model, we are using `ner_posology_large` and setting NerConverter's WhiteList `['DRUG']` in order to get only drug entities.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

## to get drugs
drugs_ner_ing = MedicalNerModel.pretrained("ner_posology_large", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_drug")

drugs_ner_converter_ing = NerConverter() \
    .setInputCols(["sentence", "token", "ner_drug"]) \
    .setOutputCol("ner_chunk")\
    .setWhiteList(["DRUG"])
  
drugs_c2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc") 

sbert_embedder_ing = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["ner_chunk_doc"])\
    .setOutputCol("sentence_embeddings")

rxnorm_resolver = SentenceEntityResolverModel.pretrained("demo_sbiobertresolve_rxnorm","en", "clinical/models")\
    .setInputCols(["ner_chunk", "sentence_embeddings"]) \
    .setOutputCol("rxnorm_code")\
    .setDistanceFunction("EUCLIDEAN")
    

pipeline_rxnorm_ingredient = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        drugs_ner_ing,
        drugs_ner_converter_ing, 
        drugs_c2doc, 
        sbert_embedder_ing,
        rxnorm_resolver])

data_ner = spark.createDataFrame([['']]).toDF("text")
rxnorm_model = pipeline_rxnorm_ingredient.fit(data_ner)

# COMMAND ----------

# MAGIC %md
# MAGIC **Visualize Drug Entities** 
# MAGIC 
# MAGIC Now we will visualize a sample text with `NerVisualizer`.

# COMMAND ----------

sample_text = df.select("text").collect()[1]

print(sample_text)

# COMMAND ----------

# MAGIC %md
# MAGIC `NerVisualizer` woks with Lightpipeline, so we will create a `light_model` with our `rxnorm_model`.

# COMMAND ----------

light_model =  LightPipeline(rxnorm_model)

ann_text = light_model.fullAnnotate(sample_text)[0]
print(ann_text.keys())

# COMMAND ----------

#Creating the vizualizer 
from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

# Change color of an entity label
visualiser.set_label_colors({'DRUG':'#008080'})
ner_vis = visualiser.display(ann_text, label_col='ner_chunk',return_html=True)

#Displaying the vizualizer 
displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC **Checking all posology entities `DRUG`, `FREQUENCY`, `DURATION`, `STRENGTH`, `FORM`, `DOSAGE` and `ROUTE` by using `ner_posology_greedy` model without  setting a `WhiteList`.**
# MAGIC 
# MAGIC **So, if you want to find just the posology entities in greedy form, here is the pipeline you can use.**

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

posology_ner_greedy = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_greedy")

ner_converter_greedy = NerConverter()\
    .setInputCols(["sentence","token","ner_greedy"])\
    .setOutputCol("ner_chunk_greedy")

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    posology_ner_greedy,
    ner_converter_greedy])

empty_data = spark.createDataFrame([[""]]).toDF("text")
posology_greedy_model = nlpPipeline.fit(empty_data)

# COMMAND ----------

light_model =  LightPipeline(posology_greedy_model)
light_result = light_model.fullAnnotate(sample_text)[0]

# COMMAND ----------

#Creating the vizualizer 
from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

# Change color of an entity label
# visualiser.set_label_colors({'DRUG':'#008080'})
ner_vis = visualiser.display(light_result, label_col='ner_chunk_greedy',return_html=True)

#Displaying the vizualizer 
displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC **Now we will transform our dataframe by usig `rxnorm_model` and get RxNORM codes of the entities.**

# COMMAND ----------

rxnorm_code_res = rxnorm_model.transform(df) 

# COMMAND ----------

# MAGIC %md
# MAGIC We are getting selecting the columns which we need and converting to Pandas DataFrame. The values in `all_codes` and `resolitions` columns are seperated by ":::" and we are converting these columns to lists.

# COMMAND ----------

rxnorm_res = rxnorm_code_res.select("path", F.explode(F.arrays_zip( 'ner_chunk.result', 'rxnorm_code.result', 'rxnorm_code.metadata')).alias("cols"))\
                            .select("path", F.expr("cols['0']").alias("drug_chunk"),
                                            F.expr("cols['1']").alias("rxnorm_code"),
                                            F.expr("cols['2']['confidence']").alias("confidence"),
                                            F.expr("cols['2']['all_k_results']").alias("all_codes"),
                                            F.expr("cols['2']['all_k_resolutions']").alias("resolutions")).toPandas()


codes = []
resolutions = []

for code, resolution in zip(rxnorm_res['all_codes'], rxnorm_res['resolutions']):
    
    codes.append(code.split(':::'))
    resolutions.append(resolution.split(':::'))
    
  
rxnorm_res['all_codes'] = codes  
rxnorm_res['resolutions'] = resolutions
rxnorm_res['drugs'] = rxnorm_res['resolutions'].apply(lambda x : x[0])

# COMMAND ----------

rxnorm_res.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1. Analyze drug usage patterns for inventory management and reporting
# MAGIC 
# MAGIC We are checking how many times any drug are encountered in the documents.

# COMMAND ----------

rxnorm_res.head()

# COMMAND ----------

print(rxnorm_res.drugs.value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2. Analyze most frequent used drugs for different cancer patients
# MAGIC 
# MAGIC For each drug, we are counting the unique documents encountered.

# COMMAND ----------

unique_drugs = rxnorm_res.groupby(["path","drugs"]).count().reset_index()[["path","drugs"]]	

plt.figure(figsize=(20,8))
plt.xticks(rotation=90)

sns.countplot(unique_drugs.drugs, order=unique_drugs.drugs.value_counts().head(30).index, palette="Set2")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Get Timeline Using RE Models
# MAGIC 
# MAGIC We will create a relation extration model to identify temporal relationships among clinical events by using pretrained **RelationExtractionModel** `re_temporal_events_clinical`.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencerDL_hc = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos_tags")

events_ner_tagger = MedicalNerModel()\
    .pretrained("ner_events_clinical", "en", "clinical/models")\
    .setInputCols("sentence", "token", "embeddings")\
    .setOutputCol("ner_tags")  

ner_chunker = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner_tags"])\
    .setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel.pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos_tags", "token"])\
    .setOutputCol("dependencies")

clinical_re_Model = RelationExtractionModel()\
    .pretrained("re_temporal_events_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\
    .setPredictionThreshold(0.9)

pipeline = Pipeline(stages=[
  documentAssembler,
  sentencerDL_hc,
  tokenizer, 
  word_embeddings, 
  pos_tagger, 
  events_ner_tagger,
  ner_chunker,
  dependency_parser,
  clinical_re_Model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = pipeline.fit(empty_data)

# COMMAND ----------

# MAGIC %md
# MAGIC **Selecting the columns we need and transforming to Pandas Dataframe.**

# COMMAND ----------

temporal_re_df = model.transform(df)

# COMMAND ----------

temporal_re_pd_df = temporal_re_df.select("path", F.explode(F.arrays_zip('relations.result', 'relations.metadata')).alias("cols"))\
                                  .select("path",
                                          F.expr("cols['0']").alias("relation"),
                                          F.expr("cols['1']['entity1']").alias('entity1'),
                                          F.expr("cols['1']['chunk1']").alias('chunk1'),
                                          F.expr("cols['1']['entity2']").alias('entity2'),
                                          F.expr("cols['1']['chunk2']").alias('chunk2'),
                                          F.expr("cols['1']['confidence']").alias('confidence')
                                         ).toPandas()

# COMMAND ----------

temporal_re_pd_df.head(40)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Analyze the Relations Between Body Parts and Procedures
# MAGIC 
# MAGIC We will create a relation extration model to identify relationships between body parts and problem entities by using pretrained **RelationExtractionModel** `re_bodypart_problem`.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencerDL_hc = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") 

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos_tags")

ner_tagger = MedicalNerModel() \
    .pretrained("jsl_ner_wip_greedy_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_tags")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner_tags"]) \
    .setOutputCol("ner_chunks_re")\
    .setWhiteList(['Internal_organ_or_component', 'Problem', 'Procedure'])

dependency_parser = DependencyParserModel() \
    .pretrained("dependency_conllu", "en") \
    .setInputCols(["sentence", "pos_tags", "token"]) \
    .setOutputCol("dependencies")

re_model = RelationExtractionModel()\
    .pretrained('re_bodypart_problem', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks_re", "dependencies"]) \
    .setOutputCol("relations")

bodypart_re_pipeline = Pipeline(stages=[documentAssembler, 
                            sentencerDL_hc, 
                            tokenizer, 
                            pos_tagger, 
                            word_embeddings, 
                            ner_tagger, 
                            ner_converter,
                            dependency_parser,
                            re_model])


empty_data = spark.createDataFrame([['']]).toDF("text")
bodypart_re_model = bodypart_re_pipeline.fit(empty_data)


# COMMAND ----------

bodypart_re_df = bodypart_re_model.transform(df)

# COMMAND ----------

bodypart_relation = bodypart_re_df.select("path", F.explode(F.arrays_zip('relations.result', 'relations.metadata')).alias("cols"))\
                                  .select("path",
                                          F.expr("cols['0']").alias("relation"),
                                          F.expr("cols['1']['entity1']").alias('entity1'),
                                          F.expr("cols['1']['chunk1']").alias('chunk1'),
                                          F.expr("cols['1']['entity2']").alias('entity2'),
                                          F.expr("cols['1']['chunk2']").alias('chunk2')
                                          ).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1. Extract internal bodypart and procedure relations
# MAGIC 
# MAGIC We are filtering the dataframe to select rows with following conditions to see the relations between different entities.
# MAGIC * `entity1 != entity2'`

# COMMAND ----------

bodypart_relation = bodypart_relation[bodypart_relation.entity1!=bodypart_relation.entity2].drop_duplicates()
bodypart_relation

# COMMAND ----------

# MAGIC %md
# MAGIC **We can see the procedures applied to internal organs**

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Get Procedure codes from notes
# MAGIC 
# MAGIC We will create a new pipeline to get procedure codes. As NER model, we are using `jsl_ner_wip_greedy_clinical` and setting NerConverter's WhiteList `['Procedure']` in order to get only drug entities.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") 

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

proc_ner = MedicalNerModel.pretrained("jsl_ner_wip_greedy_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_proc")

proc_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner_proc"]) \
    .setOutputCol("ner_chunk")\
    .setWhiteList(['Procedure'])

proc_c2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc") 

proc_sbert_embedder = BertSentenceEmbeddings\
    .pretrained("sbiobert_base_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["ner_chunk_doc"])\
    .setOutputCol("sentence_embeddings")

cpt_resolver = SentenceEntityResolverModel.pretrained("demo_sbiobertresolve_cpt","en", "clinical/models")\
    .setInputCols(["ner_chunk", "sentence_embeddings"]) \
    .setOutputCol("cpt_code")\
    .setDistanceFunction("EUCLIDEAN")

bert_pipeline_cpt = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        proc_ner,
        proc_ner_converter, 
        proc_c2doc, 
        proc_sbert_embedder,
        cpt_resolver])
empty_data = spark.createDataFrame([['']]).toDF("text")
cpt_model = bert_pipeline_cpt.fit(empty_data)

# COMMAND ----------

cpt_model_res = cpt_model.transform(df)

# COMMAND ----------

cpt_res = cpt_model_res.select("path", F.explode(F.arrays_zip( 'ner_chunk.result', 'ner_chunk.metadata', 'cpt_code.result', 'cpt_code.metadata')).alias("cols"))\
                          .select("path", F.expr("cols['0']").alias("chunks"),
                                          F.expr("cols['1']['entity']").alias("entity"),
                                          F.expr("cols['2']").alias("cpt_code"),
                                          F.expr("cols['3']['confidence']").alias("confidence"),
                                          F.expr("cols['3']['all_k_results']").alias("all_codes"),
                                          F.expr("cols['3']['all_k_resolutions']").alias("resolutions")).toPandas()

codes = []
resolutions = []

for code, resolution in zip(cpt_res['all_codes'], cpt_res['resolutions']):
    
    codes.append(code.split(':::'))
    resolutions.append(resolution.split(':::'))
    
  
cpt_res['all_codes'] = codes  
cpt_res['resolutions'] = resolutions
cpt_res['cpt'] = cpt_res['resolutions'].apply(lambda x : x[0])

# COMMAND ----------

cpt_res.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1. See most common procedures being performed
# MAGIC 
# MAGIC Let's count the number of each procedures and plot it.

# COMMAND ----------

#top 20
cpt_res['cpt'].value_counts().reset_index().head(20)

# COMMAND ----------

plt.figure(figsize=(20,8),dpi=100)
cpt_res['cpt'].value_counts().head(20).plot.bar()
plt.title('Most Common Procedures')
plt.xlabel("Procedure")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Get Assertion Status of Cancer Entities
# MAGIC 
# MAGIC We will create a new pipeline to get assertion status of cancer entities procedure codes. As NER model, we are using `jsl_ner_wip_greedy_clinical`.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

# Cancer
bionlp_ner = MedicalNerModel.pretrained("ner_bionlp", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("bionlp_ner")

bionlp_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "bionlp_ner"]) \
    .setOutputCol("bionlp_ner_chunk")\
    .setWhiteList(["Cancer"])

# Clinical Terminology
jsl_ner = MedicalNerModel.pretrained("jsl_ner_wip_greedy_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("jsl_ner")

jsl_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "jsl_ner"]) \
    .setOutputCol("jsl_ner_chunk")\
    .setWhiteList(["Oncological", "Symptom"])

chunk_merger = ChunkMergeApproach()\
    .setInputCols('bionlp_ner_chunk', "jsl_ner_chunk")\
    .setOutputCol('final_ner_chunk')

cancer_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
    .setInputCols(["sentence", "final_ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")


assertion_pipeline = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        bionlp_ner,
        bionlp_ner_converter,
        jsl_ner,
        jsl_ner_converter,
        chunk_merger,
        cancer_assertion
    ])
empty_data = spark.createDataFrame([['']]).toDF("text")
assertion_model = assertion_pipeline.fit(empty_data)

# COMMAND ----------

assertion_res = assertion_model.transform(df)

# COMMAND ----------

assertion_df = assertion_res.select("path", F.explode(F.arrays_zip('final_ner_chunk.result', 'final_ner_chunk.metadata', 'assertion.result')).alias("cols"))\
                            .select("path", F.expr("cols['0']").alias("chunk"),
                                            F.expr("cols['1']['entity']").alias("entity"),
                                            F.expr("cols['2']").alias("assertion")).toPandas()

# COMMAND ----------

assertion_df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC **We can find the number of family members of cancer patients with cancer or symptoms**

# COMMAND ----------

print("Number of family members have cancer or symptoms: ", len(assertion_df[assertion_df.assertion=="associated_with_someone_else"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Use Case: Finding assertion status of the most common symptoms 
# MAGIC 
# MAGIC We will check if the symptom is absent or present.

# COMMAND ----------

assertion_df = assertion_df.drop_duplicates()
assertion_symptom = assertion_df[(assertion_df.assertion.isin(['present', 'absent'])) & (assertion_df.entity=="Symptom")]
most_common_symptoms = assertion_symptom.groupby(['path', 'assertion', 'chunk']).count().reset_index().chunk.value_counts().index[:20]

# COMMAND ----------

assertion_symptom[assertion_symptom.chunk.isin(most_common_symptoms)]

# COMMAND ----------

assertion_df["assertion"].value_counts()

# COMMAND ----------

 assertion_df.groupby(["path","assertion"]).count().reset_index()

# COMMAND ----------

assertion_symptom[assertion_symptom.chunk.isin(most_common_symptoms)]

# COMMAND ----------

plt.figure(figsize=(20,6), dpi=150)
sns.countplot(x='chunk', data=assertion_symptom[assertion_symptom.chunk.isin(most_common_symptoms)], hue='assertion', order=most_common_symptoms )
plt.xticks(rotation = 90)
plt.title("Assertion Status Of Most Common Symptoms")
plt.xlabel("Symptom")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **End Of Notebook**
