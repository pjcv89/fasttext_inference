from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import argparse
import os


def is_valid_path(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


parser = argparse.ArgumentParser()
parser.add_argument('--use-arrow', action='store_true', help="whether use pandas_udf via Arrow instead of standard udf")
parser.add_argument('--multi-pred', action='store_true', help="whether multiple predictions must be returned")
parser.add_argument('--input-file', action='store', help="name of input text file",
                    type=lambda x: is_valid_path(parser, x))
parser.add_argument('--output-file', action='store', help="name of output file",
                    type=lambda x: is_valid_path(parser, x))
args = parser.parse_args()

if __name__ == "__main__":
    spark = SparkSession.builder.master("local[4]").appName('mysession').getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    
    spark.sparkContext.addFile('models/ft_tuned.ftz')
    spark.sparkContext.addPyFile('./classifier.py')
    
    df_input = spark.read.parquet(args.input_file)
    
    if args.use_arrow:
        print('Using pandas_udf via Arrow\n')
        from classifier import make_pandas_udf
        udf_predict = make_pandas_udf(multi_prediction=args.multi_pred, rowwise=True)
    else:
        print('Using standard udf\n')
        from classifier import make_udf
        udf_predict = make_udf(multi_prediction=args.multi_pred)

    df_output = df_input.withColumn("category", udf_predict(col("input")))
    df_output.printSchema()
    df_output.sample(False, .10).show(10, False)

    df_output.write.mode('overwrite').parquet(args.output_file)
