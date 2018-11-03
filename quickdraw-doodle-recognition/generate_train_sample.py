import os
import argparse

from common import pack_example

from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, rand, rank
from pyspark.sql.types import StructType, StructField, StringType


def write_partitions(df, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    def write_partition(i, iterator):
        with open(f'{output_dir}/part-{i}.bin', 'wb') as fout:
            for row in iterator:
                pack_example(row, fout)
        return []

    df.rdd.mapPartitionsWithIndex(write_partition, True).collect()


def generate_train_sample(spark, train_examples_per_class, val_examples_per_class, random_seed):
    input_columns = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
    output_columns = ['key_id', 'drawing', 'word']

    # Read train_simplified folder.
    input_dir = os.path.abspath('data/train_simplified/*.csv')
    input_schema = StructType([StructField(c, StringType()) for c in input_columns])
    input_df = spark.read.csv(input_dir, schema=input_schema, header=True).select(output_columns)

    # Generate a sample of examples_per_class examples from each class.
    examples_per_class = train_examples_per_class + val_examples_per_class
    window = Window.partitionBy('word').orderBy('rand')
    sample_df = input_df \
        .withColumn('rand', rand(seed=random_seed)) \
        .withColumn('rank', rank().over(window)) \
        .filter(col('rank') <= examples_per_class) \
        .persist(StorageLevel.DISK_ONLY)

    train_df = sample_df \
        .filter(col('rank') <= train_examples_per_class) \
        .repartition(100, 'rand') \
        .sortWithinPartitions('rand') \
        .select(output_columns)
    train_dir = os.path.abspath(f'data/train_simplified_sample/{random_seed}/train')
    write_partitions(train_df, train_dir)

    val_df = sample_df \
        .filter(col('rank') > train_examples_per_class) \
        .repartition(10, 'rand') \
        .sortWithinPartitions('rand') \
        .select(output_columns)
    val_dir = os.path.abspath(f'data/train_simplified_sample/{random_seed}/val')
    write_partitions(val_df, val_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_examples_per_class', type=int)
    parser.add_argument('val_examples_per_class', type=int)
    parser.add_argument('random_seed', type=int)
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()
    generate_train_sample(
        spark,
        args.train_examples_per_class,
        args.val_examples_per_class,
        args.random_seed)
