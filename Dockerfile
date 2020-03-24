FROM continuumio/miniconda3
MAINTAINER Pablo Campos Viana

RUN apt-get update \
	&& apt-get -y install gcc g++ make cmake

RUN conda update -n base -c defaults conda
RUN conda install -c anaconda openjdk pip -y
RUN pip install numpy pandas fasttext pyarrow==0.14.1 pyspark jupyter 

ENV JAVA_HOME=/opt/conda
ENV PYSPARK_PYTHON=/opt/conda/bin/python 
ENV SPARK_HOME=/opt/conda/lib/python3.7/site-packages/pyspark

RUN chmod u+x /opt/conda/lib/python3.7/site-packages/pyspark/bin/*

RUN ln -s -f /opt/conda/bin/python /usr/bin/python \
	&& alias pyspark='/opt/conda/bin/pyspark' \
	&& alias spark-submit='/opt/conda/bin/spark-submit'

WORKDIR /fasttext_inference

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888","--allow-root"]

