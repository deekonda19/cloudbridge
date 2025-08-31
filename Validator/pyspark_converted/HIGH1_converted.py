from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, StructType, StructField, TextType, LongType, DoubleType, TimestampType
from pyspark.sql.window import Window

# Read Customers
def read_customers(spark):
    customers = spark.read().format("text").option("header", "true").load()
    return customers

# Read Products
def read_products(spark):
    products = spark.read().format("text").option("header", "true").load()
    return products

# Sort Customers
def sort_customers(spark, customers):
    customers = customers.withColumn("sort_key", udf(lambda x: x[0].split(",")[1]))
    customers = customers.withColumn("sort_key", udf(lambda x: x[1].split(",")[1]))
    return customers

# Sort Products
def sort_products(spark, products):
    products = products.withColumn("sort_key", udf(lambda x: x[0].split(",")[1]))
    products = products.withColumn("sort_key", udf(lambda x: x[1].split(",")[1]))
    return products

# Merge Join
def merge_join(spark, customers, products):
    customers = customers.withColumn("customer_id", udf(lambda x: x[0].split(",")[1]))
    products = products.withColumn("product_id", udf(lambda x: x[0].split(",")[1]))
    return customers, products

# Write Merged Data
def write_merged_data(spark, merged_data):
    merged_data = merged_data.withColumn("customer_id", udf(lambda x: x[0].split(",")[1]))
    merged_data = merged_data.withColumn("product_id", udf(lambda x: x[1].split(",")[1]))
    return merged_data