from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, StructType, StructField, TextType, Row
from pyspark.sql.window import Window

# Read People
read_people = spark.read.format("text") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("path/to/file.txt")

# Group By
group_by = read_people.selectExpr("name") \
        .groupBy(col("name"))

# Sort Counts
sort_counts = group_by.agg(udf(lambda x: len(x), StringType())) \
        .selectExpr("name", "count") \
        .groupBy("name") \
        .agg(udf(lambda x: sum(col("count") for col in x), StringType()))

# Write Job Title Counts
write_job_title_counts = sort_counts.selectExpr("name", "count") \
        .groupBy("name") \
        .agg(udf(lambda x: len(x), StringType())) \
        .selectExpr("name", "count") \
        .groupBy("name") \
        .agg(udf(lambda x: sum(col("count") for col in x), StringType())) \
        .writeStream \
        .format("text") \
        .option("header", "true") \
        .option("checkpointLocation", "path/to/checkpoint")