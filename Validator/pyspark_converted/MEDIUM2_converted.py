from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, StructType, StructField, TextType, IntegerType, LongType, TimestampType
from pyspark.sql.window import Window

# Define a UDF to convert the 'name' column to a string
def udf_name(row):
    return row['name']

# Create a DataFrame from the given steps
df = spark.createDataFrame([
    {'name': 'Read People', 'type': 'TextFileInput'},
    {'name': 'Select Values', 'type': 'SelectValues'},
    {'name': 'Sort Rows', 'type': 'SortRows'},
    {'name': 'Write Sorted People', 'type': 'TextFileOutput'}
], [
    StringType(),
    StructType([
        StructField('name', StringType()),
        StructField('type', StringType()),
        StructField('values', StringType())
    ]),
    StructType([
        StructField('name', StringType()),
        StructField('type', StringType()),
        StructField('values', StringType())
    ]),
    StructType([
        StructField('name', StringType()),
        StructField('type', StringType()),
        StructField('values', StringType())
    ]),
    StructType([
        StructField('name', StringType()),
        StructField('type', StringType()),
        StructField('values', StringType())
    ]),
    StructType([
        StructField('name', StringType()),
        StructField('type', StringType()),
        StructField('values', StringType())
    ]),
    StructType([
        StructField('name', StringType()),
        StructField('type', StringType()),
        StructField('values', StringType())
    ]),
    StructType([
        StructField('name', StringType()),
        StructField('type', StringType()),
        StructField('values', StringType())
    ]),
    StructType([
        StructField('name', StringType()),
        StructField('type', StringType()),
        StructField('values', StringType())
    ]),
    StructType([
        StructField('name', StringType()),
        StructField('type', StringType()),
        StructField('values', StringType())
    ]),
    StructType([