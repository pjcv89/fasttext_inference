# fastText inference with PySpark 

## Overview

This repo. shows how to perform  inference using a [fastText](https://fasttext.cc/) model with PySpark via user defined functions (**UDF's**), and via RDD's **mapPartitions**.

For this illustrative example, we consider we have used the [stacksample](https://www.kaggle.com/stackoverflow/stacksample)  data for the use case where given (short) text of questions titles, we want to predict their most probable tags. For more info. regarding the data, the processing, and the training stage, you can refer to this [tutorial](https://github.com/pjcv89/AutoTag/).

Here we assume we have:

- An already trained fastText model file.
- A Parquet file with only text input, with already proper processing and cleaning, ready for inference.

Please note that you can substitute these with another model file and another Parquet file, regarding other use cases. In short, the aim of the repo. is to show alternatives to perform scalable inference on the Parquet file using a fastText model file through PySpark.

## Requirements

Make sure you have [Docker](https://www.docker.com/get-started) installed in your machine. The Docker base image is [this](https://hub.docker.com/r/continuumio/miniconda3), which has a standard installation of [miniconda](https://docs.conda.io/en/latest/miniconda.html) (based on Python 3.7). As you will be able to see from the Dockerfile, the following tools and libraries are installed when building the image.

Tools:

- [gcc](https://gcc.gnu.org/)
- g++
- [make](https://www.gnu.org/software/make/)
- [cmake](https://cmake.org/)
- [openjdk 1.8.0](https://anaconda.org/anaconda/openjdk)

Python libraries:

- [NumPy](https://pypi.org/project/numpy/)
- [Pandas](https://pypi.org/project/pandas/)
- [fastText](https://pypi.org/project/fasttext/)
- [PySpark 2.4.5](https://pypi.org/project/pyspark/)
- [PyArrow 0.14.1](https://pypi.org/project/pyarrow/0.14.1/)
- [Jupyter](https://pypi.org/project/jupyter/)

## Usage 

Once you have chosen a proper working directory on your local machine, clone this repo. and go inside the repo. folder.

```bash
git clone https://github.com/pjcv89/fasttext_inference.git
cd fasttext_inference 
```

Now, build the image using the provided Dockerfile, and give it a name and a tag. For example:

```bash
docker image build -t fasttext_inference:0.1 .
```

Once you have built the image, you can use the container in two ways. 

### a) Using the command line

You can execute: 
```bash
docker run --name inference -v $PWD:/fasttext_inference -it --entrypoint=/bin/bash fasttext_inference:0.1
```
In this mode, you will be able to invoke `spark-submit` to execute the `inference.py` and `inference_mapp.py` scripts from the command line and from the current working directory.

The purpose of both scripts is to take a Parquet file ready for inference specified via the `--input-file` argument and produce an output Parquet file with predictions, whose location needs to be specified via the `--output-file` argument.

In both scripts we can choose between:
- Retrieving single predictions (the prediction with highest probability for each instance, whenever its probability is above a certain threshold). 
-  Retrieving multiple predictions (those predictions whose probabilities are above a certain threshold, retrieving at most *k* predictions for each instance) via the `--multi-pred` flag. 

Note that in either case, if the threshold condition is not met, a `null` value is returned. Currently the default values of the threshold and *k* are set to `threshold=0.10` and `k=3`.

##### Using UDF's approach  (`inference.py` script)

In this script we can choose between:

- Using Spark UDF's (one-row-at-a-time execution of UDF, this is the default behavior in the script).
- Using Pandas UDF's for PySpark (execution of UDF by chunks of `pandas.Series`), which are built on top of [Arrow](https://arrow.apache.org/), via the `--use-arrow` flag. 

For example, launching the following job will use the standard UDF's approach and retrieve single predictions: 

```bash
spark-submit inference.py --input-file data/input.parquet --output-file data/output.parquet
```

While launching the following job will use Pandas UDF's approach and retrieve multiple predictions instead:

```bash
spark-submit inference.py --input-file data/input.parquet --output-file data/output.parquet --use-arrow --multi-pred
```

##### Using RDD's mapPartitions approach  (`inference_mapp.py` script)

This approach is inspired by [this discussion](https://www.facebook.com/groups/1174547215919768/?comment_id=2913166652057807&comment_tracking=%7B%22tn%22%3A%22R%22%7D&post_id=2913128998728239) and follows a different logic by using the powerful Spark's [mapPartitions](https://medium.com/@ajaygupta.hbti/apache-spark-mappartitions-a-powerful-narrow-data-transformation-d635964526d6) transformation.

For example, launching the following job will use the RDD's mapPartitions approach and retrieve multiple predictions:

```bash
spark-submit inference_mapp.py --input-file data/input.parquet --output-file data/output.parquet --multi-pred
```

### b) Using the Jupyter notebook

You can execute:
```bash
docker run --name inference -p 8080:8888 -v $PWD:/fasttext_inference fasttext_inference:0.1
```
Jupyter will be launched and you can go to [http://localhost:8080/](http://localhost:8080/). You should copy the token displayed in the command line and paste it in the jupyter welcome page. You will be able to see the files contained in this repo., including the notebook, which you can open to start executing code.

## Files and folders

The following files are provided:

- ``Dockerfile``: The Dockerfile to build the image.
- ``classifier.py``: Python file with required functions to construct the UDF's and called by the ``inference.py`` Python script.
- ``inference.py``: Python script relative to the UDF's approach, to be executed via `spark-submit`.
- ``inference_mapp.py``: Python script relative to the RDD's mapPartitions approach, to be executed via `spark-submit`.

The following folders are present:
- `/data`: It contains the  `test` and `test_unlabeled` text files, where the latter is just the unlabeled version of the former. It also contains `/input.parquet` folder where the input Parquet file built from `test_unlabeled` and ready for inference is located, and `/output.parquet`folder where the output Parquet file  with predictions will be persisted after executing any of the Python scripts.
- `/models`: It contains the already trained fastText model, called `ft_tuned.ftz`
- `/notebooks`: It contains the following notebooks, which contain some prototyping code for the Python scripts and some performance tests. Names are self-explanatory.

0. ``00_Input_Data.ipynb``: Notebook that shows how the input Parquet file was generated. You can view it [here](https://github.com/pjcv89/fasttext_inference/blob/master/notebooks/00_Input_Data.ipynb).
1. ``01_Standard_UDFs.ipynb``: View it [here](https://nbviewer.jupyter.org/github/pjcv89/fasttext_inference/blob/master/notebooks/01_Standard_UDFs.ipynb).
2. ``02_Pandas_UDFs.ipynb``: View it [here](https://nbviewer.jupyter.org/github/pjcv89/fasttext_inference/blob/master/notebooks/02_Pandas_UDFs.ipynb).
3. ``03_RDDs_mapPartitions.ipynb``: View it [here](https://nbviewer.jupyter.org/github/pjcv89/fasttext_inference/blob/master/notebooks/03_RDDs_mapPartitions.ipynb).

#### **BONUS**: Using RDD's pipe approach 

This approach is also inspired by [this discussion](https://www.facebook.com/groups/1174547215919768/?comment_id=2913166652057807&comment_tracking=%7B%22tn%22%3A%22R%22%7D&post_id=2913128998728239) and uses Spark's [pipe](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.pipe) method to call external processes. In this case, we use fastText CLI tool to get predictions using a shell script to be called within the `pipe` method. 

The `/pipe` folder includes the following files:

- `04_RDDs_pipe.ipynb`: Notebook that shows how to carry out this approach. View it [here](https://nbviewer.jupyter.org/github/pjcv89/fasttext_inference/blob/master/pipe/04_RDDs_pipe.ipynb).
- `install_fasttext.sh`: Shell script to build fastText from source and install CLI tool. Used in the notebook.
- `get_predictions.sh`: Shell script to be called within the `pipe` method. Used in the notebook.

Please refer to the following posts:

1. [Spark pipe: A one-pipe problem](https://bit.ly/2xMMtqj)
2. [Pipe in Spark](http://blog.madhukaraphatak.com/pipe-in-spark/)

## Important note: Distributed settings

Please note that all examples here use Spark's local mode and client mode. For the UDF's approach shown here, in order to make the model file and the Python module available among workers, we have included the following lines in the `inference.py` script:

```python
spark.sparkContext.addFile('models/ft_tuned.ftz')
spark.sparkContext.addPyFile('./classifier.py')
```

However, in a distributed setting and in cluster mode, we would need to distribute these files across nodes using the `files` and `--py-files` options instead. See this [question](https://stackoverflow.com/questions/38879478/sparkcontext-addfile-vs-spark-submit-files).

## Resources

### Engineering

- Ideas used in this repo.
1. [Classifying text with fastText in pySpark](https://www.futurice.com/blog/classifying-text-with-fasttext-in-pyspark)
2. [Prediction at Scale with scikit-learn and PySpark Pandas UDF's](https://medium.com/civis-analytics/prediction-at-scale-with-scikit-learn-and-pyspark-pandas-udfs-51d5ebfb2cd8)
3. [Introducing Pandas UDF's for PySpark](https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html)
4. [Pandas user-defined functions](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html#pandas-user-defined-functions)
5. [PySpark Usage Guide for Pandas with Apache Arrow](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html)

### Science

- fastText related papers:

1. [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
2. [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
3. [FastText.zip: Compressing text classification models](https://arxiv.org/abs/1612.03651)
4. [Misspelling Oblivious Word Embeddings](https://arxiv.org/abs/1905.09755)

- Papers about techniques used in fastText to improve scalability and training time:

1. [Hierarchical softmax based on the Huffman coding tree](https://arxiv.org/abs/1301.3781)
2. [The hashing trick](https://arxiv.org/abs/0902.2206)