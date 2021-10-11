# Databricks notebook source
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
# MAGIC     JSL_OCR_LICENSE=iii
# MAGIC     ```
# MAGIC 
# MAGIC 3. Download the followings with AWS CLI to your local computer
# MAGIC 
# MAGIC      `$ aws s3 cp --region us-east-2 s3://pypi.johnsnowlabs.com/$jsl_secret/spark-nlp-jsl-$jsl_version.jar spark-nlp-jsl-$jsl_version.jar`
# MAGIC 
# MAGIC     `$ aws s3 cp --region us-east-2 s3://pypi.johnsnowlabs.com/$jsl_secret/spark-nlp-jsl/spark_nlp_jsl-$jsl_version-py3-none-any.whl spark_nlp_jsl-$jsl_version-py3-none-any.whl` 
# MAGIC     
# MAGIC     `$ aws s3 cp --region us-east-2 s3://pypi.johnsnowlabs.com/$jsl_ocr_secret/jars/spark-ocr-assembly-$ocr_version-spark30.jar spark-ocr-assembly-$ocr_version-spark30.jar`
# MAGIC     
# MAGIC     `$ aws s3 cp --region us-east-2 s3://pypi.johnsnowlabs.com/$jsl_ocr_secret/spark-ocr/spark_ocr-$ocr_version.spark30-py3-none-any.whl spark_ocr-$ocr_version.spark30-py3-none-any.whl`
# MAGIC     
# MAGIC 
# MAGIC 4. In `Libraries` tab inside your cluster:
# MAGIC 
# MAGIC  - Install New -> PyPI -> `spark-nlp==$public_version` -> Install
# MAGIC  - Install New -> Maven -> Coordinates -> `com.johnsnowlabs.nlp:spark-nlp_2.12:$public_version` -> Install
# MAGIC 
# MAGIC  - add following jars for the Healthcare library that you downloaded above:
# MAGIC  
# MAGIC         - Install New -> Python Whl -> upload `spark_nlp_jsl-$jsl_version-py3-none-any.whl`
# MAGIC 
# MAGIC         - Install New -> Jar -> upload `spark-nlp-jsl-$jsl_version.jar`
# MAGIC         
# MAGIC         - Install New -> Python Whl -> upload `spark_ocr-$ocr_version.spark30-py3-none-any.whl`
# MAGIC 
# MAGIC         - Install New -> Jar -> upload `spark-ocr-assembly-$ocr_version-spark30.jar`
# MAGIC 
# MAGIC 5. Now you can attach your notebook to the cluster and use Spark NLP!
# MAGIC 
# MAGIC For more information, see 
# MAGIC 
# MAGIC   https://nlp.johnsnowlabs.com/docs/en/install#databricks-support
# MAGIC 
# MAGIC   https://nlp.johnsnowlabs.com/docs/en/licensed_install#install-spark-nlp-for-healthcare-on-databricks

# COMMAND ----------

# MAGIC %md
# MAGIC # Spark OCR in Healtcare
# MAGIC 
# MAGIC Spark OCR is a commercial extension of Spark NLP for optical character recognition from images, scanned PDF documents, Microsoft DOCX and DICOM files. 
# MAGIC 
# MAGIC In this notebook we will:
# MAGIC   - Parsing the Files through OCR.
# MAGIC   - Extract PHI entites from extracted texts.
# MAGIC   - Hide PHI entites and get an obfucated versions of pdf files.
# MAGIC   - Hide PHI entities on original image.
# MAGIC   - Extract text from some Dicom images.

# COMMAND ----------

import os
import json
import string
#import sys
#import base64
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

import sparkocr
from sparkocr.transformers import *
from sparkocr.utils import *
from sparkocr.enums import *

from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from sparknlp.training import CoNLL

import matplotlib.pyplot as plt

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 100)  
pd.set_option('display.expand_frame_repr', False)

spark.sql("set spark.sql.legacy.allowUntypedScalaUDF=true")

print('sparknlp.version : ',sparknlp.version())
print('sparknlp_jsl.version : ',sparknlp_jsl.version())
print('sparkocr : ',sparkocr.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC **Reading PDF files **
# MAGIC - We have some PDF files in delta tables and we are starting by reading them. We can use `spark.read` and load them by `load` function.

# COMMAND ----------

# MAGIC %sh
# MAGIC for i in {0..1}
# MAGIC do
# MAGIC   wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/ocr/MT_0$i.pdf
# MAGIC done

# COMMAND ----------

pdfs = spark.read.format("binaryFile").load("file:/databricks/driver/MT_*")

print("Number of files in the folder : ", pdfs.count())

# COMMAND ----------

pdfs.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Parsing the Files through OCR

# COMMAND ----------

# MAGIC %md
# MAGIC - The pdf files can have more than one page. We will transform the document in to images per page. Than we can run OCR to get text. 
# MAGIC - We are using `PdfToImage()` to render PDF to images and `ImageToText()` to runs OCR for each images.

# COMMAND ----------

# Transform PDF document to images per page
pdf_to_image = PdfToImage()\
      .setInputCol("content")\
      .setOutputCol("image")

# Run OCR
ocr = ImageToText()\
      .setInputCol("image")\
      .setOutputCol("text")\
      .setConfidenceThreshold(65)\
      .setIgnoreResolution(False)

ocr_pipeline = PipelineModel(stages=[
    pdf_to_image,
    ocr
])

# COMMAND ----------

# MAGIC %md
# MAGIC - Now, we can transform the `pdfs` with our pipeline.

# COMMAND ----------

ocr_result = ocr_pipeline.transform(pdfs)

# COMMAND ----------

# MAGIC %md
# MAGIC - After transforming we get following columns :
# MAGIC 
# MAGIC   - path
# MAGIC   - modificationTime
# MAGIC   - length
# MAGIC   - image
# MAGIC   - total_pages
# MAGIC   - pagenum
# MAGIC   - documentnum
# MAGIC   - confidence
# MAGIC   - exception
# MAGIC   - text
# MAGIC   - positions

# COMMAND ----------

ocr_result.select('modificationTime', 'length', 'total_pages', 'pagenum', 'documentnum', 'confidence', 'exception').show(truncate=False)

# COMMAND ----------

ocr_result.select('path', 'image', 'text', 'positions').show(truncate=30)

# COMMAND ----------

# MAGIC %md
# MAGIC - Now, we have our pdf files in text format and as image. 
# MAGIC 
# MAGIC - In `sparkocr`library, we have `display_image` function to display any image. Images are stored in the `image` column of `results`.  Let's see the image and the text.

# COMMAND ----------

import matplotlib.pyplot as plt

img = ocr_result.collect()[0].image
img_pil = to_pil_image(img, img.mode)

plt.figure(figsize=(24,16))
plt.imshow(img_pil, cmap='gray')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Let's see extracted text which is stored in `'text'` column as a list. Each line is is an item in this list, so we can join them and see the whole page.

# COMMAND ----------

print("\n".join([row.text for row in ocr_result.select("text").collect()[0:1]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1. Skew Correction
# MAGIC 
# MAGIC In some images, there may be some skewness and this reduces acuracy of the extracted text. Spark OCR has `ImageSkewCorrector` which detects skew of the image and rotates it.

# COMMAND ----------

# Image skew corrector 
pdf_to_image = PdfToImage()\
      .setInputCol("content")\
      .setOutputCol("image")

skew_corrector = ImageSkewCorrector()\
      .setInputCol("image")\
      .setOutputCol("corrected_image")\
      .setAutomaticSkewCorrection(True)

ocr = ImageToText()\
      .setInputCol("corrected_image")\
      .setOutputCol("text")\
      .setConfidenceThreshold(65)\
      .setIgnoreResolution(False)

ocr_skew_corrected = PipelineModel(stages=[
    pdf_to_image,
    skew_corrector,
    ocr
])

# COMMAND ----------

ocr_skew_corrected_result = ocr_skew_corrected.transform(pdfs).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see the results after the skew correction.

# COMMAND ----------

print("Original Images")
ocr_result.filter(ocr_result.path=="file:/databricks/driver/MT_01.pdf").select('path', 'confidence').show(truncate=False)

print("Skew Corrected Images")
ocr_skew_corrected_result.filter(ocr_skew_corrected_result.path=="file:/databricks/driver/MT_01.pdf").select('path', 'confidence').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC After skew correction, confidence is increased from %48.3 to % %66.5. Let's display the corrected image and the original image side by side.

# COMMAND ----------

img_orig = ocr_skew_corrected_result.select("image").collect()[3].image
img_corrected = ocr_skew_corrected_result.select("corrected_image").collect()[3].corrected_image

img_pil_orig = to_pil_image(img_orig, img_orig.mode)
img_pil_corrected = to_pil_image(img_corrected, img_corrected.mode)

plt.figure(figsize=(24,16))
plt.subplot(1, 2, 1)
plt.imshow(img_pil_orig, cmap='gray')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(img_pil_corrected, cmap='gray')
plt.title("Skew Corrected")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2. Image Processing
# MAGIC 
# MAGIC * After reading pdf files, we can process on images to increase the confidency.
# MAGIC 
# MAGIC * By **`ImageAdaptiveThresholding`**, we can compute a threshold mask image based on local pixel neighborhood and apply it to image. 
# MAGIC 
# MAGIC * Another method which we can add to pipeline is applying morphological operations. We can use **`ImageMorphologyOperation`** which support:
# MAGIC   - Erosion
# MAGIC   - Dilation
# MAGIC   - Opening
# MAGIC   - Closing   
# MAGIC 
# MAGIC * To remove remove background objects **`ImageRemoveObjects`** can be used.
# MAGIC 
# MAGIC * We will add **`ImageLayoutAnalyzer`** to pipeline, to analyze the image and determine the regions of text.

# COMMAND ----------

from sparkocr.enums import *

# Read binary as image
pdf_to_image = PdfToImage()\
  .setInputCol("content")\
  .setOutputCol("image")\
  .setResolution(400)

# Correcting the skewness
skew_corrector = ImageSkewCorrector()\
      .setInputCol("image")\
      .setOutputCol("skew_corrected_image")\
      .setAutomaticSkewCorrection(True)

# Binarize using adaptive tresholding
binarizer = ImageAdaptiveThresholding()\
  .setInputCol("skew_corrected_image")\
  .setOutputCol("binarized_image")\
  .setBlockSize(91)\
  .setOffset(50)

# Apply morphology opening
opening = ImageMorphologyOperation()\
  .setKernelShape(KernelShape.SQUARE)\
  .setOperation(MorphologyOperationType.OPENING)\
  .setKernelSize(3)\
  .setInputCol("binarized_image")\
  .setOutputCol("opening_image")

# Remove small objects
remove_objects = ImageRemoveObjects()\
  .setInputCol("opening_image")\
  .setOutputCol("corrected_image")\
  .setMinSizeObject(130)

# Run OCR for corrected image
ocr_corrected = ImageToText()\
  .setInputCol("corrected_image")\
  .setOutputCol("corrected_text")\
  .setPageIteratorLevel(PageIteratorLevel.SYMBOL) \
  .setPageSegMode(PageSegmentationMode.SPARSE_TEXT) \
  .setConfidenceThreshold(70)

# OCR pipeline
image_pipeline = PipelineModel(stages=[
    pdf_to_image,
    skew_corrector,
    binarizer,
    opening,
    remove_objects,
    ocr_corrected
])

# COMMAND ----------

result_processed = image_pipeline.transform(pdfs).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see the original image and corrected image.

# COMMAND ----------

img_orig = result_processed.select("image").collect()[3].image
img_corrected = result_processed.select("corrected_image").collect()[3].corrected_image

img_pil_orig = to_pil_image(img_orig, img_orig.mode)
img_pil_corrected = to_pil_image(img_corrected, img_corrected.mode)

plt.figure(figsize=(24,16))
plt.subplot(1, 2, 1)
plt.imshow(img_pil_orig, cmap='gray')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(img_pil_corrected, cmap='gray')
plt.title("Skew Corrected")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC After processing, we have cleaner image. And confidence is increased to %97

# COMMAND ----------

print("Original Images")
ocr_result.filter(ocr_result.path=="file:/databricks/driver/MT_01.pdf").select('path', 'confidence').show(truncate=False)

print("Skew Corrected Images")
ocr_skew_corrected_result.filter(ocr_skew_corrected_result.path=="file:/databricks/driver/MT_01.pdf").select('path', 'confidence').show(truncate=False)

print("Corrected Images")
result_processed.filter(result_processed.path=="file:/databricks/driver/MT_01.pdf").select('path', 'confidence').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Extracting the PHI Entites
# MAGIC 
# MAGIC Now Let's create a clinical NER pipeline and see which entities we have. We will use `sentence_detector_dl_healthcare` to detect sentences and get entities by using [`jsl_ner_wip_clinical`](https://nlp.johnsnowlabs.com/2021/03/31/jsl_ner_wip_clinical_en.html) in `MedicalNerModel`.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
  .setInputCols(["document"]) \
  .setOutputCol("sentence")

tokenizer = Tokenizer()\
  .setInputCols(["sentence"])\
  .setOutputCol("token")\

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")

ner = MedicalNerModel.pretrained("jsl_ner_wip_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")

# COMMAND ----------

ner_pipeline = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        ner,
        ner_converter])

empty_data = spark.createDataFrame([['']]).toDF("text")
ner_model = ner_pipeline.fit(empty_data)

# COMMAND ----------

ner_results = ner_model.transform(ocr_result)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will visualize a sample text with `NerVisualizer`.
# MAGIC 
# MAGIC `NerVisualizer` woks with Lightpipeline, so we will create a `light_model` with our ner_model.

# COMMAND ----------

sample_text = ocr_result.select("text").collect()[1].text

# COMMAND ----------

print(sample_text)

# COMMAND ----------

light_model =  LightPipeline(ner_model)
 
ann_text = light_model.fullAnnotate(sample_text)[0]

# COMMAND ----------

# MAGIC %md
# MAGIC `fullAnnotate` method returns the results as a dictionary. But the dictionary stored in a list. So we can reach to the dict by adding `[0]` to the end of the annotated text.
# MAGIC 
# MAGIC We can get some columns and transform them to a Pandas dataframe.

# COMMAND ----------

chunks = []
entities = []
sentence= []
begin = []
end = []

for n in ann_text['ner_chunk']:
        
    begin.append(n.begin)
    end.append(n.end)
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    sentence.append(n.metadata['sentence'])
    
    
import pandas as pd

df = pd.DataFrame({'chunks':chunks, 'begin': begin, 'end':end, 
                   'sentence_id':sentence, 'entities':entities})

df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC We can visualise the annotated text by `display` method of `NerVisualizer()`

# COMMAND ----------

from sparknlp_display import NerVisualizer
 
visualiser = NerVisualizer()

ner_vis = visualiser.display(ann_text, label_col='ner_chunk',return_html=True)
 
displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Hiding the PHI Entities
# MAGIC 
# MAGIC In our documents we have some fields which we want to hide. To do it, we will use deidentification model. It identifies instances of protected health information in text documents, and it can either obfuscate them (e.g., replacing names with different, fake names) or mask them.

# COMMAND ----------

def deidentification_nlp_pipeline(input_column, prefix = ""):
    document_assembler = DocumentAssembler() \
        .setInputCol(input_column) \
        .setOutputCol(prefix + "document")

    # Sentence Detector annotator, processes various sentences per line
    sentence_detector = SentenceDetector() \
        .setInputCols([prefix + "document"]) \
        .setOutputCol(prefix + "sentence")

    tokenizer = Tokenizer() \
        .setInputCols([prefix + "sentence"]) \
        .setOutputCol(prefix + "token")

    # Clinical word embeddings
    word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
        .setInputCols([prefix + "sentence", prefix + "token"]) \
        .setOutputCol(prefix + "embeddings")
    # NER model trained on i2b2 (sampled from MIMIC) dataset
    clinical_ner = MedicalNerModel.pretrained("ner_deid_large", "en", "clinical/models") \
        .setInputCols([prefix + "sentence", prefix + "token", prefix + "embeddings"]) \
        .setOutputCol(prefix + "ner")

    custom_ner_converter = NerConverterInternal() \
        .setInputCols([prefix + "sentence", prefix + "token", prefix + "ner"]) \
        .setOutputCol(prefix + "ner_chunk")

    nlp_pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            word_embeddings,
            clinical_ner,
            custom_ner_converter
        ])
    empty_data = spark.createDataFrame([[""]]).toDF(input_column)
    nlp_model = nlp_pipeline.fit(empty_data)
    return nlp_model

# COMMAND ----------

obfuscation = DeIdentification()\
      .setInputCols(["sentence", "token", "ner_chunk"]) \
      .setOutputCol("deidentified") \
      .setMode("obfuscate")\
      .setObfuscateRefSource("faker")\
      .setObfuscateDate(True)

obfuscation_pipeline = Pipeline(stages=[
        image_pipeline,                        
        deidentification_nlp_pipeline(input_column="corrected_text"),
        obfuscation

    ])

# COMMAND ----------

from pyspark.sql.types import BinaryType

empty_data = spark.createDataFrame([['']]).toDF("path")
empty_data = empty_data.withColumn('content', empty_data.path.cast(BinaryType()))

#empty_data = spark.createDataFrame([[""]]).toDF('content')
obfuscation_model = obfuscation_pipeline.fit(empty_data)

# COMMAND ----------

obfuscation_result = obfuscation_model.transform(pdfs).cache()

# COMMAND ----------

result_df = obfuscation_result.select(F.explode(F.arrays_zip('token.result', 'ner.result')).alias("cols")) \
                              .select(F.expr("cols['0']").alias("token"),
                                      F.expr("cols['1']").alias("ner_label"))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's count the number of entities we want to deidentificate and then see them.

# COMMAND ----------

result_df.select("token", "ner_label").groupBy('ner_label').count().orderBy('count', ascending=False).show(truncate=False)

# COMMAND ----------

obfuscation_result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
                  .select(F.expr("cols['0']").alias("chunk"),
                          F.expr("cols['1']['entity']").alias("ner_label")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC In deidentified column, entities like date and name are replaced by fake identities. Let's see some of them.

# COMMAND ----------

obfusated_text_df = obfuscation_result.select('path', F.explode(F.arrays_zip('sentence.result', 'deidentified.result')).alias("cols")) \
                                      .select('path', F.expr("cols['0']").alias("sentence"), F.expr("cols['1']").alias("deidentified")).toPandas()

# COMMAND ----------

obfusated_text_df.iloc[[4]]

# COMMAND ----------

print("*" * 30)
print("Original Text")
print("*" * 30)
print(obfusated_text_df.iloc[4]['sentence'][:100])

print("*" * 30)
print("Obfusated Text")
print("*" * 30)

print(obfusated_text_df.iloc[4]['deidentified'][:100])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Getting Obfuscated Version of Each File
# MAGIC Now we have obfuscated version of each file in dataframe. Each page is in diiferent page. Let's merge and save the files as txt.

# COMMAND ----------

obfusated_text_df['Deidintified_Test'] = obfusated_text_df.groupby('path').deidentified.transform((lambda x: ''.join(x)))
obfuscated_versions = obfusated_text_df[['path', 'Deidintified_Test']].drop_duplicates()

obfuscated_versions

# COMMAND ----------

#Writing txt versions
for index, row in obfuscated_versions.iterrows():
  with open(row.path.split("/")[-1].replace('pdf', 'txt'), 'w') as txt:
    txt.write(row.Deidintified_Test)

# COMMAND ----------

# MAGIC %md
# MAGIC We have written the txt files with the same name with .txt extension. Let's read and see the a file.

# COMMAND ----------

from os import listdir

filenames = listdir(".")
text_files = [ filename for filename in filenames if filename.endswith('.txt') ]
text_files

# COMMAND ----------

with open(text_files[0], 'r') as txt:
  print(txt.read())

# COMMAND ----------

# MAGIC %md
# MAGIC ##  5. Image Deidentifier 
# MAGIC Above, we replaced some entities with fake entities. This time we will hide these entities with a blank line on the the original image.

# COMMAND ----------

# Read binary as image
pdf_to_image = PdfToImage()\
  .setInputCol("content")\
  .setOutputCol("image_raw")\
  .setResolution(400)

# Extract text from image
ocr = ImageToText() \
    .setInputCol("image_raw") \
    .setOutputCol("text") \
    .setIgnoreResolution(False) \
    .setPageIteratorLevel(PageIteratorLevel.SYMBOL) \
    .setPageSegMode(PageSegmentationMode.SPARSE_TEXT) \
    .setConfidenceThreshold(65)

# Found coordinates of sensitive data
position_finder = PositionFinder() \
    .setInputCols("ner_chunk") \
    .setOutputCol("coordinates") \
    .setPageMatrixCol("positions") \
    .setMatchingWindow(1000) \
    .setPadding(1)\
    .setWindowPageTolerance(False)

# Draw filled rectangle for hide sensitive data
drawRegions = ImageDrawRegions()  \
    .setInputCol("image_raw")  \
    .setInputRegionsCol("coordinates")  \
    .setOutputCol("image_with_regions")  \
    .setFilledRect(True) \
    .setRectColor(Color.black)
    
# OCR pipeline
deid_pipeline = Pipeline(stages=[
    pdf_to_image,
    ocr,
    deidentification_nlp_pipeline(input_column="text"),
    position_finder,
    drawRegions
])

empty_data = spark.createDataFrame([['']]).toDF("path")
empty_data = empty_data.withColumn('content', empty_data.path.cast(BinaryType()))
model = deid_pipeline.fit(pdfs)

# COMMAND ----------

deid_result = model.transform(pdfs).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets show the PHI entities detected on pdf files.**

# COMMAND ----------

deid_result.select("ner_chunk.result").show(50, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **Here is an example of deidentified images**

# COMMAND ----------

r = deid_result.select("image_raw", "image_with_regions").collect()[3]
img_orig = r.image_raw
img_deid = r.image_with_regions

img_pil_orig = to_pil_image(img_orig, img_orig.mode)
img_pil_deid = to_pil_image(img_deid, img_deid.mode)

plt.figure(figsize=(24,16))
plt.subplot(1, 2, 1)
plt.imshow(img_pil_orig, cmap='gray')
plt.title('original')
plt.subplot(1, 2, 2)
plt.imshow(img_pil_deid, cmap='gray')
plt.title("de-id'd")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Extracting Text from DICOM images
# MAGIC 
# MAGIC We have 3 dicom samples and read them from spark-ocr-workshop repo. We will extract text from them. Let's start by reading them.

# COMMAND ----------

# MAGIC %sh
# MAGIC for i in {1..3}
# MAGIC do
# MAGIC   wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-ocr-workshop/master/jupyter/data/dicom/deidentify-medical-$i.dcm
# MAGIC done

# COMMAND ----------

dicom_df = spark.read.format("binaryFile").load("file:/databricks/driver/deidentify-medical*.dcm")

# COMMAND ----------

# MAGIC %md
# MAGIC We can convert Dicom images to image by **`DicomToImage`**.

# COMMAND ----------

dicomToImage = DicomToImage() \
  .setInputCol("content") \
  .setOutputCol("image") \
  .setMetadataCol("meta")

data = dicomToImage.transform(dicom_df)

for r in data.select("image").collect():
    img = r.image
    img_pil = to_pil_image(img, img.mode)
    plt.figure(figsize=(24,16))
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil, cmap='gray')

# COMMAND ----------

# MAGIC %md
# MAGIC And finally, let's extract the text from these images.

# COMMAND ----------

# Extract text from image
ocr = ImageToText() \
    .setInputCol("image") \
    .setOutputCol("text") \
    .setIgnoreResolution(False) \
    .setOcrParams(["preserve_interword_spaces=0"])


print("\n".join([row.text for row in ocr.transform(data).select("text").collect()]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Dicom Image Deidentifier

# COMMAND ----------

dicom_to_image = DicomToImage() \
    .setInputCol("content") \
    .setOutputCol("image_raw") \
    .setMetadataCol("metadata") \
    .setDeIdentifyMetadata(True)

ocr = ImageToText() \
    .setInputCol("image_raw") \
    .setOutputCol("text") \
    .setIgnoreResolution(False) \
    .setOcrParams(["preserve_interword_spaces=0"])

# Found coordinates of sensitive data
position_finder = PositionFinder() \
    .setInputCols("ner_chunk") \
    .setOutputCol("coordinates") \
    .setPageMatrixCol("positions") \
    .setMatchingWindow(1000) \
    .setPadding(1)

# Draw filled rectangle for hide sensitive data
drawRegions = ImageDrawRegions()  \
    .setInputCol("image_raw")  \
    .setInputRegionsCol("coordinates")  \
    .setOutputCol("image_with_regions")  \
    .setFilledRect(True) \
    .setRectColor(Color.black)\
    .setLineWidth(10)

# Store image back to Dicom document
imageToDicom = ImageToDicom() \
    .setInputCol("image_with_regions") \
    .setOutputCol("dicom") 


pipeline = Pipeline(stages=[
    dicom_to_image,
    ocr,
    deidentification_nlp_pipeline(input_column="text"),
    position_finder,
    drawRegions,
    imageToDicom
])

# COMMAND ----------

deid_dicom_df = pipeline.fit(dicom_df).transform(dicom_df).cache()

# COMMAND ----------

deid_dicom_df.show(truncate=70)

# COMMAND ----------

for r in deid_dicom_df.select("dicom", "path").collect():
    path, name = os.path.split(r.path)
    filename_split = os.path.splitext(name)
    file_name = os.path.join("/dbfs/", filename_split[0]+".dcm")
    print(f"Storing to {file_name}")
    with open(file_name, "wb") as file:
        file.write(r.dicom)

# COMMAND ----------

# MAGIC %md
# MAGIC ** Display Deidentified Image and Deidentified metadata**

# COMMAND ----------

dicom_gen_df = spark.read.format("binaryFile").load("file:/dbfs/*1.dcm")
de_dicom_gen_df = DicomToImage().transform(dicom_gen_df)
for r in de_dicom_gen_df.select("image").collect():
    img = r.image
    img_pil = to_pil_image(img, img.mode)
    plt.figure(figsize=(24,16))
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil, cmap='gray')
