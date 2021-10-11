# Databricks notebook source
# MAGIC %md
# MAGIC # Create a Database of Oncological Entities Based on Unstructured Notes
# MAGIC In this notebook, we create a database of entities based on extracted terms from the notes in previous notebooks. 
# MAGIC This database can be used for dashboarding using [Databricks SQL](https://databricks.com/product/databricks-sql).

# COMMAND ----------

delta_path='/FileStore/HLS/nlp/delta/jsl/'

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Temporary Views

# COMMAND ----------

spark.read.load(f"{delta_path}/silver/icd10-hcc-df").createOrReplaceTempView('icd10Hcc')
spark.read.load(f"{delta_path}/gold/best-icd-mapped").createOrReplaceTempView('bestIcdMapped')
spark.read.load(f"{delta_path}/gold/rxnorm-res-cleaned").createOrReplaceTempView('rxnormRes')
spark.read.load(f"{delta_path}/silver/rxnorm-code-greedy-res").createOrReplaceTempView('rxnormCodeGreedy')
spark.read.load(f"{delta_path}/silver/temporal-re").createOrReplaceTempView('temporalRe')
spark.read.load(f"{delta_path}/silver/bodypart-relationships").createOrReplaceTempView('bodypartRelationships')
spark.read.load(f"{delta_path}/silver/cpt").createOrReplaceTempView('cpt')
spark.read.load(f"{delta_path}/silver/assertion").createOrReplaceTempView('assertion')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the Database

# COMMAND ----------

database_name='jsl_onc'
DatabaseName=''.join([st.capitalize() for st in database_name.split('_')])
database_path=f"{delta_path}tables/{database_name}"
print(f"{DatabaseName} database tables will be stored in {database_path}")

# COMMAND ----------

sql(f"DROP DATABASE IF EXISTS {DatabaseName} CASCADE;")
sql(f"CREATE DATABASE IF NOT EXISTS {DatabaseName} LOCATION '{database_path}'")
sql(f"USE {DatabaseName};")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Tables

# COMMAND ----------

CREATE OR REPLACE TABLE Rxnorm_Res AS (
  select md5(path) as note_id,path,confidence, drug_chunk,rxnorm_code,drugs as drug from rxnormRes
);

# COMMAND ----------

CREATE OR REPLACE TABLE CPT AS (
  select md5(path) as note_id, path, confidence, chunks, entity,cpt_code,cpt
from cpt)

# COMMAND ----------

CREATE OR REPLACE TABLE ASSERTION AS (
  select md5(path) as note_id, path, chunk, entity,assertion from assertion
)

# COMMAND ----------

CREATE OR REPLACE TABLE TEMPORAL_RE AS (
  select md5(path) as note_id, * from temporalRe
)

# COMMAND ----------

CREATE OR REPLACE TABLE BEST_ICD AS (
  select * from bestIcdMapped
)

# COMMAND ----------

CREATE OR REPLACE TABLE ICD10_HCC AS (
  select md5(path) as note_id, path, confidence, final_chunk, entity,icd10_code,icd_codes_names,icd_code_billable
  from icd10Hcc
)

# COMMAND ----------

select * from ICD10_HCC

# COMMAND ----------

# MAGIC %md
# MAGIC ## License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disclaimers
# MAGIC Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.
