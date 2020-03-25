from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import col, concat_ws
from classifier import make_udf, make_pandas_udf
import argparse
import os


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg
    
def is_valid_path(parser, arg):
    parent_path = os.path.dirname(arg)
    if not os.path.exists(parent_path):
        parser.error("The directory %s does not exist!" % arg)
    else:
        return arg
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--use-arrow', action='store_true',help="whether use pandas_udf via Arrow instead of standard udf")
parser.add_argument('--multi-pred', action='store_true',help="whether multiple predictions must be returned")
parser.add_argument('--input-file',action='store',help="name of input text file",type=lambda x: is_valid_file(parser, x))
parser.add_argument('--output-file',action='store',help="name of output file",type=lambda x: is_valid_path(parser, x))
args = parser.parse_args()


if __name__ == "__main__":
    spark = SparkSession.builder.appName('mysession').getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    schema = StructType([StructField("input", StringType())])
    df_input = spark.read.csv(args.input_file, header=False, schema=schema)

    if args.use_arrow:
        print('Using pandas_udf via Arrow\n')
        udf_predict = make_pandas_udf(multi_prediction=args.multi_pred)
    else:
        print('Using standard udf\n')
        udf_predict = make_udf(multi_prediction=args.multi_pred)

    df_output = df_input.withColumn("category",udf_predict(col("input")))
    df_output.sample(False,.10).show(10,False)
    
    if args.multi_pred:
        df_output = df_output.withColumn('category', concat_ws('|', 'category'))
    df_output.write.option("header","true").option("sep",",").mode("overwrite").csv(args.output_file)