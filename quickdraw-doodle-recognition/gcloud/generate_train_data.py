import os

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.sql.types import StructType, StructField, StringType

import common
import drawing


def write_partitions(df, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    def write_partition(i, iterator):
        examples = list(iterator)
        rng = np.random.RandomState(common.RANDOM_SEED + i)
        rng.shuffle(examples)
        bin_file = os.path.join(output_dir, 'part-{}.bin'.format(i + 1))
        with open(bin_file, 'wb') as fout:
            for example in examples:
                image = drawing.process_drawing(example['drawing'], common.IMAGE_SIZE)
                label = common.WORD2LABEL[example['word']]
                common.pack_example(image, label, fout)
        return []

    df.rdd.mapPartitionsWithIndex(write_partition, True).collect()


def generate_train_data(spark, num_partitions):
    input_dir = os.path.abspath('input/train_simplified/*.csv')
    input_columns = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
    input_schema = StructType([StructField(c, StringType()) for c in input_columns])
    input_df = spark.read.csv(input_dir, schema=input_schema, header=True).select('drawing', 'word')

    output_df = input_df.repartition(num_partitions, rand(seed=common.RANDOM_SEED))
    write_partitions(output_df, 'output/train_simplified/')


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    generate_train_data(spark, num_partitions=2 * common.NUM_CPU)
