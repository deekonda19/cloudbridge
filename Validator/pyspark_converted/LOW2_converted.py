from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, StructType, TextType, Row
from pyspark.sql.window import Window

# Define a function to extract the lead ID from the text column
def extract_lead_id(row):
    return row['Lead ID']

# Create a schema for the data
schema = StructType([
    Column("name", StringType()),
    Column("type", StringType()),
    Column("text", StringType())
])

# Define a function to extract the lead ID from the text column
def extract_lead_id(row):
    return row['Lead ID']

# Create a dataframe from the text columns
df = spark.createDataFrame([
    ('name', 'Read Leads'),
    ('type', 'TextFileInput'),
    ('text', 'Lead ID'),
    ('name', 'Write Leads'),
    ('type', 'TextFileOutput')
], schema)

# Define a function to extract the lead ID from the text column
def extract_lead_id(row):
    return row['Lead ID']

# Create a dataframe from the text columns
df2 = spark.createDataFrame([
    ('name', 'Read Leads'),
    ('type', 'TextFileInput'),
    ('text', 'Lead ID'),
    ('name', 'Write Leads'),
    ('type', 'TextFileOutput')
], schema)

# Define a function to extract the lead ID from the text column
def extract_lead_id(row):
    return row['Lead ID']

# Create a dataframe from the text columns
df3 = spark.createDataFrame([
    ('name', 'Read Leads'),
    ('type', 'TextFileInput'),
    ('text', 'Lead ID'),
    ('name', 'Write Leads'),
    ('type', 'TextFileOutput')
], schema)

# Define a function to extract the lead ID from the text column
def extract_lead_id(row):
    return row['Lead ID']

# Create a dataframe from the text columns
df4 = spark.createDataFrame([
    ('name', 'Read Leads'),
    ('type', 'TextFileInput'),