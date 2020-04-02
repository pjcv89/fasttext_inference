from pyspark.sql import SparkSession
from pyspark.sql import Row
import fasttext
import argparse
import os


def is_valid_path(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


def fn_partition(iterator, multi_prediction=False):
    def get_predictions(sentence, threshold=0.10, k=3):
        """
        Note: This is the same function as in classifier.py module!
        """
        labels, probs = model.predict(sentence.lower(), k=k)
        output = list(
            map(lambda y: y[0].replace("__label__", ""), filter(lambda x: x[1] >= threshold, zip(labels, probs))))
        if len(output) == 0:
            return None
        else:
            return output[0] if k == 1 else output

    model = fasttext.load_model('models/ft_tuned.ftz')

    for record in iterator:
        if not multi_prediction:
            yield Row(category=get_predictions(record['input'], k=1), input=record['input'])
        else:
            yield Row(category=get_predictions(record['input'], k=3), input=record['input'])


parser = argparse.ArgumentParser()
parser.add_argument('--multi-pred', action='store_true', help="whether multiple predictions must be returned")
parser.add_argument('--input-file', action='store', help="name of input text file",
                    type=lambda x: is_valid_path(parser, x))
parser.add_argument('--output-file', action='store', help="name of output file",
                    type=lambda x: is_valid_path(parser, x))
args = parser.parse_args()

if __name__ == "__main__":
    spark = SparkSession.builder.appName('mysession').getOrCreate()
    df_input = spark.read.parquet(args.input_file)

    print('Using RDDs mapPartitions\n')
    df_output = df_input.rdd.mapPartitions(lambda partition: fn_partition(partition, args.multi_pred)).toDF()
    df_output.printSchema()
    df_output.sample(False, .10).show(10, False)

    df_output.write.mode('overwrite').parquet(args.output_file)
