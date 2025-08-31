from pyspark.sql.functions import col, row_number
from pyspark.sql.types import StringType, IntegerType, LongType
from pyspark.sql.window import Window

# Read Products
readProducts = spark.read.format("text") \
    .option("header", "true") \
    .load("path/to/products.txt")

# Write Products
writeProducts = spark.write.mode("overwrite").format("text") \
    .option("header", "false") \
    .option("partitionColumns", "name") \
    .option("partitionValues", "value") \
    .option("partitionNumber", "1") \
    .option("partitionType", "first") \
    .option("partitionSortOrder", "true") \
    .mode("append").format("text") \
    .option("header", "false") \
    .option("partitionColumns", "name") \
    .option("partitionValues", "value") \
    .option("partitionNumber", "1") \
    .option("partitionType", "first") \
    .option("partitionSortOrder", "true") \
    .mode("append").format("text") \
    .option("header", "false") \
    .option("partitionColumns", "name") \
    .option("partitionValues", "value") \
    .option("partitionNumber", "1") \
    .option("partitionType", "first") \
    .option("partitionSortOrder", "true") \
    .mode("append").format("text") \
    .option("header", "false") \
    .option("partitionColumns", "name") \
    .option("partitionValues", "value") \
    .option("partitionNumber", "1") \
    .option("partitionType", "first") \
    .option("partitionSortOrder", "true") \
    .mode("append").format("text") \
    .option("header", "false") \
    .option("partitionColumns", "name")