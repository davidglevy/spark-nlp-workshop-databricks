# Databricks notebook source
slides_html="""

<iframe src="https://docs.google.com/presentation/d/1yR3oBKg8vvwKjvj4WWezf5ygJweo8rWuklD7IF4uVX0/embed?start=true&loop=true&delayms=4000" frameborder="0" width="900" height="560" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
"""
displayHTML(slides_html)

# COMMAND ----------

# MAGIC %md
# MAGIC # Spark OCR in Healthcare
# MAGIC 
# MAGIC Spark OCR is a commercial extension of Spark NLP for optical character recognition from images, scanned PDF documents, Microsoft DOCX and DICOM files. 
# MAGIC 
# MAGIC In this notebook we will:
# MAGIC   - Parsing the Files through OCR.
# MAGIC   - Extract PHI entites from extracted texts.
# MAGIC   - Hide PHI entites and get an obfucated versions of pdf files.
# MAGIC   - Hide PHI entities on original image.
# MAGIC   - Extract text from some Dicom images.
# MAGIC   - Hide PHI entities on Dicom images.

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
# MAGIC 
# MAGIC If you have large datasets, you can take your data into delta table and read from there to create a dataframe by using this script:  
# MAGIC 
# MAGIC ```
# MAGIC pdfs = spark.read.format("delta").load("/mnt/delta/ocr_samples")
# MAGIC 
# MAGIC print("Number of files in the folder : ", pdfs.count())
# MAGIC 
# MAGIC ```

# COMMAND ----------

# MAGIC %sh
# MAGIC for i in {0..3}
# MAGIC do
# MAGIC   wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/data/ocr/MT_OCR_0$i.pdf -P /dbfs/FileStore/HLS/nlp/
# MAGIC done

# COMMAND ----------

file_path='dbfs:/FileStore/HLS/nlp/'
pdfs = spark.read.format("binaryFile").load(f'{file_path}MT_OCR*.pdf').sort('path')
print("Number of files in the folder : ", pdfs.count())

# COMMAND ----------

display(pdfs)

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

display(
  ocr_result.select('modificationTime', 'length', 'total_pages', 'pagenum', 'documentnum', 'confidence', 'exception')
)

# COMMAND ----------

display(ocr_result.select('path', 'image', 'text', 'positions'))

# COMMAND ----------

# MAGIC %md
# MAGIC - Now, we have our pdf files in text format and as image. 
# MAGIC 
# MAGIC - Let's see the images and the text.

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

# DBTITLE 1,Original Images
display(ocr_result.filter(ocr_result.path==f"{file_path}MT_OCR_01.pdf").select('path', 'confidence'))

# COMMAND ----------

# DBTITLE 1,Skew Corrected Images
display(ocr_skew_corrected_result.filter(ocr_skew_corrected_result.path==f"{file_path}MT_OCR_01.pdf").select('path', 'confidence'))

# COMMAND ----------

# MAGIC %md
# MAGIC After skew correction, confidence is increased from %48.3 to % %66.5. Let's display the corrected image and the original image side by side.

# COMMAND ----------

img_orig = ocr_skew_corrected_result.select("image").collect()[1].image
img_corrected = ocr_skew_corrected_result.select("corrected_image").collect()[1].corrected_image

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


ocr_corrected = ImageToText()\
  .setInputCol("corrected_image")\
  .setOutputCol("corrected_text")\
  .setPageIteratorLevel(PageIteratorLevel.SYMBOL) \
  .setPageSegMode(PageSegmentationMode.SPARSE_TEXT) \
  .setConfidenceThreshold(65)

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

img_orig = result_processed.select("image").collect()[1].image
img_corrected = result_processed.select("corrected_image").collect()[1].corrected_image

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
display(ocr_result.filter(ocr_result.path==f"{file_path}MT_OCR_01.pdf").select('confidence'))

print("Skew Corrected Images")
display(ocr_skew_corrected_result.filter(ocr_skew_corrected_result.path==f"{file_path}MT_OCR_01.pdf").select('confidence'))

print("Corrected Images")
display(result_processed.filter(result_processed.path==f"{file_path}MT_OCR_01.pdf").select('confidence'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Extracting the Clinical Entites
# MAGIC 
# MAGIC Now Let's create a clinical NER pipeline and see which entities we have. We will use `sentence_detector_dl_healthcare` to detect sentences and get entities by using [`ner_jsl`](https://nlp.johnsnowlabs.com/2021/06/24/ner_jsl_en.html) in `MedicalNerModel`.

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

clinical_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
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
        clinical_ner,
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

sample_text = ocr_result.limit(2).select("text").collect()[0].text

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
# MAGIC ## 3. Extracting and Hiding the PHI Entities
# MAGIC 
# MAGIC In our documents we have some fields which we want to hide. To do it, we will use deidentification model. It identifies instances of protected health information in text documents, and it can either obfuscate them (e.g., replacing names with different, fake names) or mask them.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
  .setInputCol("corrected_text")\
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
 
deid_ner = MedicalNerModel.pretrained("ner_deid_generic_augmented", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
 
deid_ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")
 
deid_pipeline = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        deid_ner,
        deid_ner_converter])
 
empty_data = spark.createDataFrame([['']]).toDF("corrected_text")
deid_model = deid_pipeline.fit(empty_data)

# COMMAND ----------

# MAGIC %md
# MAGIC We created another pipeline which detects PHI entities. We will use the same text above and visualize again.

# COMMAND ----------

sample_text = ocr_result.limit(2).select("text").collect()[0].text
light_deid_model =  LightPipeline(deid_model)
 
ann_text = light_deid_model.fullAnnotate(sample_text)[0]

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
    
df = pd.DataFrame({'chunks':chunks, 'begin': begin, 'end':end, 
                   'sentence_id':sentence, 'entities':entities})
				   
visualiser = NerVisualizer()

ner_vis = visualiser.display(ann_text, label_col='ner_chunk',return_html=True)
 
displayHTML(ner_vis)

# COMMAND ----------

# Transform PDF document to images per page
pdf_to_image = PdfToImage()\
      .setInputCol("content")\
      .setOutputCol("image")

# Run OCR
ocr = ImageToText()\
      .setInputCol("image")\
      .setOutputCol("corrected_text")\
      .setConfidenceThreshold(65)\
      .setIgnoreResolution(False)

obfuscation = DeIdentification()\
      .setInputCols(["sentence", "token", "ner_chunk"]) \
      .setOutputCol("deidentified") \
      .setMode("obfuscate")\
      .setObfuscateRefSource("faker")\
      .setObfuscateDate(True)

obfuscation_pipeline = Pipeline(stages=[
        pdf_to_image,
        ocr,
        deid_pipeline,
        obfuscation
    ])

# COMMAND ----------

from pyspark.sql.types import BinaryType

empty_data = spark.createDataFrame([['']]).toDF("path")
empty_data = empty_data.withColumn('content', empty_data.path.cast(BinaryType()))

obfuscation_model = obfuscation_pipeline.fit(empty_data)

# COMMAND ----------

obfuscation_result = obfuscation_model.transform(pdfs)

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

obfusated_text_df[:20]

# COMMAND ----------

print("*" * 30)
print("Original Text")
print("*" * 30)
print(obfusated_text_df.iloc[3]['sentence'])

print("*" * 30)
print("Obfusated Text")
print("*" * 30)

print(obfusated_text_df.iloc[3]['deidentified'])


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Getting Obfuscated Version of Each File
# MAGIC Now we have obfuscated version of each file in dataframe. Each page is in diiferent page. Let's merge and save the files as txt.

# COMMAND ----------

obfusated_text_df['Deidentified_Test'] = obfusated_text_df.groupby('path').deidentified.transform((lambda x: '\n'.join(x)))
obfuscated_versions = obfusated_text_df[['path', 'Deidentified_Test']].drop_duplicates()

obfuscated_versions

# COMMAND ----------

#Writing txt versions
for index, row in obfuscated_versions.iterrows():
  with open(row.path.split("/")[-1].replace('pdf', 'txt'), 'w') as txt:
    txt.write(row.Deidentified_Test)

# COMMAND ----------

# MAGIC %md
# MAGIC We have written the txt files with the same name with .txt extension. Let's read and see the a file.

# COMMAND ----------

from os import listdir

filenames = listdir(".")
text_files = [ filename for filename in filenames if filename.endswith('.txt') ]

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

skew =ImageSkewCorrector()\
  .setInputCol("image_raw")\
  .setOutputCol("corrected_image")\
  .setAutomaticSkewCorrection(True)
    
# Extract text from image
ocr = ImageToText() \
    .setInputCol("corrected_image") \
    .setOutputCol("text") \
    .setIgnoreResolution(False) \
    .setPageIteratorLevel(PageIteratorLevel.SYMBOL) \
    .setPageSegMode(PageSegmentationMode.SPARSE_TEXT) \

image_to_hocr = ImageToHocr() \
    .setInputCol("corrected_image") \
    .setOutputCol("hocr")\
    .setIgnoreResolution(False)\
    .setPageIteratorLevel(PageIteratorLevel.TEXTLINE)\
    .setPageSegMode(PageSegmentationMode.SPARSE_TEXT)\
    .setOcrParams(['preserve_interword_spaces=1'])

# OCR pipeline
ocr_pipeline = PipelineModel(stages=[
    pdf_to_image,
    skew,
    ocr,
    image_to_hocr
])


# COMMAND ----------

ocr_result = ocr_pipeline.transform(pdfs)

# COMMAND ----------

import pandas as pd
import xml.etree.ElementTree as ET
from lxml import etree

def hocr_to_dataframe(hocr):

    with open ("demo_hocr_content.xml",'w',encoding='utf-8') as f:

        f.write(str(hocr))

    doc = etree.parse("demo_hocr_content.xml")

    words    = []
    wordConf = []
    fonts    = []
    sizes    = []
    font     = -1
    size     = -1
    
    for path in doc.xpath('//*'):
        
        try:
            if 'ocr_line' in path.values():
                a = float(path.values()[2].split(';')[3].split()[1])
                b = float(path.values()[2].split(';')[4].split()[1])
                font  = round((a+b)/2, 2)
                size = float(path.values()[2].split(';')[2].split()[1])

            if 'ocrx_word' in path.values():
                conf = [x for x in path.values() if 'x_wconf' in x][0]
                wordConf.append((conf.split('bbox ')[1].split(";")[0].split()))
                words.append(path.text)
                fonts.append(font)
                sizes.append(int(size))
        except:
            pass
    dfReturn = pd.DataFrame({'word' : words,
                             'bbox' : wordConf,
                            'borders':fonts,
                            'size':sizes})
        
    try:
        dfReturn = dfReturn[dfReturn['word'].str.strip()!=''].reset_index(drop=True)
    except:
        pass
    
    return(dfReturn)

import re

def get_token_df(text):
    
    try:
        tokens, borders = zip(*[(m.group(0), (m.start(), m.end()-1)) for m in re.finditer(r'\S+', text)])
    
        tuples = [(x, y[0], y[1]) for x,y in zip(tokens, borders)]
    except:
        tuples = [('-',0,0)]

    df = pd.DataFrame(tuples, columns=['token','start','end'])

    return df


def get_mapping(text, hocr=None):
    
    hdf = hocr_to_dataframe(hocr)

    token_df = get_token_df(text)
    
    token_df['start'] = token_df['start'].astype(int)
    
    token_df['end'] = token_df['end'].astype(int)
        
    token_df = pd.concat([hdf, token_df], axis=1)[['token','start','end','bbox','borders','size']]
    
    token_df['h'] = token_df.bbox.apply(lambda x: int(x[3])-int(x[1]) if type(x) is not float else 0)
    
    return token_df



def get_coordinates_frame(ent_dict_list, text, hocr):
    
    token_df = get_mapping(text, hocr)

    for i,ent in enumerate(ent_dict_list):

        ix = list(set(token_df[(token_df.start>=ent['begin'])|(token_df.end>=ent['begin'])].index).intersection(set(token_df[(token_df.start<=ent['end']+1)|(token_df.end<=ent['end']+1)].index)))

        coords = token_df.loc[ix,'bbox'].values

        if len(coords)>0:

            xo_list = [] 
            x1_list = []
            yo_list = []
            y1_list = []
            for box in coords:
                try:
                    xo_list.append(int(box[0]))
                    yo_list.append(int(box[1]))
                    x1_list.append(int(box[2]))
                    y1_list.append(int(box[3]))
                except:
                    xo_list.append(0)
                    yo_list.append(0)
                    x1_list.append(0)
                    y1_list.append(0)

            ent['coord'] = (min(xo_list), min(yo_list), max(x1_list), max(y1_list))
        else:

            ent['coord'] = []


    coord_df_pipe = pd.DataFrame(ent_dict_list)

    return coord_df_pipe

import matplotlib.pyplot as plt
from IPython.display import Image 
from PIL import Image, ImageFont, ImageDraw, ImageEnhance


def draw_comparisons(img_pil_orig, img_pil_deid, coord_df):

    draw = ImageDraw.Draw(img_pil_deid)

    for i,row in coord_df.iterrows():

        point = row['coord']

        draw.rectangle((row['coord'][:2], row['coord'][2:]), fill="black")

    plt.figure(figsize=(24,16))
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil_orig, cmap='gray')
    plt.title('original image')
    plt.subplot(1, 2, 2)
    plt.imshow(img_pil_deid, cmap='gray')
    plt.title("de-identified image")
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC We will use `clinical_deidentification` pretrained pipeline to deidentify the PHI entities. This pipeline can be used with any text.

# COMMAND ----------

import matplotlib.pyplot as plt
from IPython.display import Image 
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("clinical_deidentification", "en", "clinical/models")

for row in [0,2]:

  text = ocr_result.select('text').collect()[row][0]

  hocr =  ocr_result.select('hocr').collect()[row][0]

  ner_result = deid_pipeline.fullAnnotate(text)

  ent_dict_list = [{'begin':x.begin, 'end':x.end, 'chunk':x.result, 'ner_label':x.metadata['entity'], 'sentence_id':x.metadata['sentence']} for x in ner_result[0]['ner_chunk']]

  coord_df = get_coordinates_frame(ent_dict_list, text, hocr)
  
  img_deid = ocr_result.select('image_raw').collect()[row][0]
  img_pil_orig = to_pil_image(img_deid, img_deid.mode)
  img_deid = ocr_result.select('corrected_image').collect()[row][0]
  img_pil_deid = to_pil_image(img_deid, img_deid.mode)

  draw_comparisons(img_pil_orig, img_pil_deid, coord_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Our pipeline drew black strips over the PHI entities succesfully.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Extracting and Hiding PHI Entities on DICOM images
# MAGIC 
# MAGIC If you have a large dicom image dataset, you can take your dataset into delta tabel and read from there by using this script:
# MAGIC 
# MAGIC ```
# MAGIC dicom_df = spark.read.format("delta").load("/mnt/delta/dicom_samples")
# MAGIC ```

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-ocr-workshop/master/jupyter/data/dicom/deidentify-medical-2.dcm  -P /dbfs/FileStore/HLS/dicom/
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-ocr-workshop/master/jupyter/data/dicom/deidentify-brains-front-medical-3.dcm -P /dbfs/FileStore/HLS/dicom/

# COMMAND ----------

file_path='/FileStore/HLS/dicom/*.dcm'
dicom_df = spark.read.format("binaryFile").load(file_path).sort('path')

print("Number of files in the folder : ", dicom_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC We can convert Dicom images to image by **`DicomToImage`**.

# COMMAND ----------

dicomToImage = DicomToImage() \
  .setInputCol("content") \
  .setOutputCol("image") \
  .setMetadataCol("meta")

data = dicomToImage.transform(dicom_df)

display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see the images.

# COMMAND ----------

for r in data.select("image").collect():
    img = r.image
    img_pil = to_pil_image(img, img.mode)
    plt.figure(figsize=(24,16))
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil, cmap='gray')

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we will create a pipeline which converts dicom files to images, extracts text and deidentify PHI entities.

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
 
    custom_ner_converter = NerConverter() \
        .setInputCols([prefix + "sentence", prefix + "token", prefix + "ner"]) \
        .setOutputCol(prefix + "ner_chunk") \
        .setWhiteList(['NAME', 'AGE', 'CONTACT',
                   'LOCATION', 'PROFESSION', 'PERSON']) #You can set the whitelist accordingly
 
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

# Read dicom as image
dicom_to_image = DicomToImage() \
    .setInputCol("content") \
    .setOutputCol("image_raw") \
    .setMetadataCol("metadata") \
    .setDeIdentifyMetadata(True)

adaptive_thresholding = ImageAdaptiveThresholding() \
    .setInputCol("image_raw") \
    .setOutputCol("corrected_image") \
    .setBlockSize(47) \
    .setOffset(4) \
    .setKeepInput(True)

# Extract text from image
ocr = ImageToText() \
    .setInputCol("corrected_image") \
    .setOutputCol("text")

# Found coordinates of sensitive data
position_finder = PositionFinder() \
    .setInputCols("ner_chunk") \
    .setOutputCol("coordinates") \
    .setPageMatrixCol("positions") \
    .setMatchingWindow(100) \
    .setPadding(1)

# Draw filled rectangle for hide sensitive data
drawRegions = ImageDrawRegions()  \
    .setInputCol("image_raw")  \
    .setInputRegionsCol("coordinates")  \
    .setOutputCol("image_with_regions")  \
    .setFilledRect(True) \
    .setRectColor(Color.black)

# Store image back to Dicom document
imageToDicom = ImageToDicom() \
    .setInputCol("image_with_regions") \
    .setOutputCol("dicom") 
    
# OCR pipeline
deid_pipeline = PipelineModel(stages=[
    dicom_to_image,
    adaptive_thresholding,
    ocr,
    deidentification_nlp_pipeline(input_column="text"),
    position_finder,
    drawRegions,
])

# COMMAND ----------

# MAGIC %md
# MAGIC Let's transform the dataframe and see the extracted text.

# COMMAND ----------

deid_results = deid_pipeline.transform(dicom_df).cache()

display(deid_results.select("text"))

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can see original and deidentified images side by side.

# COMMAND ----------

for r in deid_results.select("image_raw", "image_with_regions").collect():
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
# MAGIC ## License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library License|Library License URL|Library Source URL|
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
# MAGIC |Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
# MAGIC |Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
# MAGIC |Matplotlib | | https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE | https://github.com/matplotlib/matplotlib|
# MAGIC |Pillow (PIL) | HPND License| https://github.com/python-pillow/Pillow/blob/master/LICENSE | https://github.com/python-pillow/Pillow/|
# MAGIC |lxml|BSD License|https://github.com/lxml/lxml/blob/master/doc/licenses/BSD.txt|https://github.com/lxml/lxml|
# MAGIC |IPython|BSD License|https://github.com/ipython/ipython/blob/master/LICENSE|https://github.com/ipython/ipython|
# MAGIC |Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
# MAGIC |Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
# MAGIC |Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
# MAGIC |Spark OCR |[Proprietary license - John Snow Labs Inc.](https://nlp.johnsnowlabs.com/docs/en/ocr) |NA|NA|
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
