from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import StringType, StructType, StructField, TextType, TimestampType
from pyspark.sql.window import Window

# Read leads from a text file
def read_leads(text_file):
    leads = spark.read.format("text").option("header", "true").load(text_file)
    return leads

# Read people from a text file
def read_people(text_file):
    people = spark.read.format("text").option("header", "true").load(text_file)
    return people

# Read organizations from a text file
def read_organizations(text_file):
    organizations = spark.read.format("text").option("header", "true").load(text_file)
    return organizations

# Sort leads by name and organization
def sort_leads(leads, people, organizations):
    sorted_leads = leads.withColumn("name", udf(lambda x: x["name"].lower())) \
                                 .withColumn("organization", udf(lambda x: x["organization"].lower())) \
                                 .selectExpr("name", "organization")
    return sorted_leads

# Sort people by name and organization
def sort_people(people, organizations):
    sorted_people = people.withColumn("name", udf(lambda x: x["name"].lower())) \
                                 .withColumn("organization", udf(lambda x: x["organization"].lower())) \
                                 .selectExpr("name", "organization")
    return sorted_people

# Sort organizations by name and organization
def sort_organizations(organizations, people):
    sorted_organizations = organizations.withColumn("name", udf(lambda x: x["name"].lower())) \
                                 .withColumn("organization", udf(lambda x: x["organization"].lower())) \
                                 .selectExpr("name", "organization")
    return sorted_organizations

# Merge leads and people
def merge_leads_and_people(leads, people, organizations):
    merged_leads = spark.sql.mergeJoin(leads, people,