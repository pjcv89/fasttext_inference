# fastText inference via PySpark 

## Overview
This tutorial shows how to perform  inference using a [fastText](https://fasttext.cc/) model via PySpark user defined functions (**UDF's**). 

For this illustrative example, we consider we have used the [stacksample](https://www.kaggle.com/stackoverflow/stacksample)  data for the use case where given (short) text of questions titles, we want to predict their most probable tags. For more info. regarding the data, the processing, and the training stage, you can refer to this [tutorial](https://github.com/pjcv89/AutoTag/).

In this tutorial we assume we have:

- An already trained fastText model.
- A test dataset with only text input, with already proper processing and cleaning, ready for inference.

Thus, the aim of the tutorial is to show a way to perform scalable inference on the test dataset using the fastText model through PySpark.

## Requirements

Make sure you have [Docker](https://www.docker.com/get-started) installed in your machine . The Docker base image is [this](https://hub.docker.com/r/continuumio/miniconda3), which has a standard installation of [miniconda](https://docs.conda.io/en/latest/miniconda.html) (based on Python 3.7). As you will be able to see from the Dockerfile, the following tools and libraries are installed when building the image.

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
In this mode, you will be able to invoke `spark-submit` to execute the `inference.py` script from the command line and from the current directory.

##### About the `inference.py` script 

The script aims to take a text file ready for inference specified via the `--input-file` argument and produce an output text file with predictions, whose location needs to be specified via the `--output-file` argument.

Additionally, we can choose:

- To retrieve single predictions (the prediction with highest probability for each instance, this is the default behavior in the script) or multiple predictions (the top-3 predictions for each instance) via the `--multi-pred` flag. 
- To use standard Spark UDF's (one-row-at-a-time execution of UDF, this is the default behavior in the script) or to use Pandas UDF's for PySpark (execution of UDF by chunks of Pandas Series), which are built on top of [Arrow](https://arrow.apache.org/), via the `--use-arrow` flag. 

For example, launching the following job will use the standard UDF's approach and retrieve single predictions: 

```bash
spark-submit inference.py --input-file data/spark_input --output-file data/spark_output
```

While launching the following job will use Pandas UDF's approach and retrieve multiple predictions instead:

```bash
spark-submit inference.py --input-file data/spark_input --output-file data/spark_output --use-arrow --multi-pred
```

### b) Using the Jupyter notebook

You can execute:
```bash
docker run --name inference -p 8080:8888 -v $PWD:/fasttext_inference fasttext_inference:0.1
```
Jupyter will be launched and you can go to [http://localhost:8080/](http://localhost:8080/). You should copy the token displayed in the command line and paste it in the jupyter welcome page. You will be able to see the files contained in this repo, including the notebook, which you can open to start executing code.

## Files and folders

The following files are provided:

- ``Dockerfile``: The Dockerfile to build the image.
- ``classifier.py``: Python file with required functions to construct the UDF's and called by the Python script.
- ``inference.py``: Python script to be executed via `spark-submit`.
- ``Prototypes_and_tests.ipynb``: The development notebook which contains some prototyping code for the Python script and some performance tests. You can view the notebook with Jupyter Notebook Viewer [here](https://nbviewer.jupyter.org/github/pjcv89/fasttext_inference/blob/master/Prototypes_and_tests.ipynb).

The following folders present:
- */data*: It contains the  `test` and `spark_input` text files. The latter is just the unlabeled version of the former, and ready for inference. It also contains the folder */spark_output* where output files will be persisted after executing the Python script.
- */models*: It contains the already trained fastText model, called `ft_tuned.ftz`

## Resources

### Engineering

- Ideas used in this repo.
1. [Classifying text with fastText in pySpark](https://www.futurice.com/blog/classifying-text-with-fasttext-in-pyspark)
2. [Prediction at Scale with scikit-learn and PySpark Pandas UDF's](https://medium.com/civis-analytics/prediction-at-scale-with-scikit-learn-and-pyspark-pandas-udfs-51d5ebfb2cd8)
3. [Introducing Pandas UDF for PySpark](https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html)
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