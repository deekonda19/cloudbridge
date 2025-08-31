from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, StructType, StructField, TextType, IntegerType, BooleanType
from pyspark.sql.window import Window

# Read Leads
def read_leads(row):
    return row[0]

# Read Organizations
def read_organizations(row):
    return row[1]# Sort Leads
def sort_leads(row):
    return row[0].cast("text") + row[1].cast("text")

# Sort Organizations
def sort_organizations(row):
    return row[0].cast("text") + row[1].cast("text")

# Merge Join
def merge_join(row1, row2):
    return row1.cast("text") + row2.cast("text")

# Filter Closed Won Leads
def filter_closed_won_leads(row):
    return row[0] == "Closed Won"

# Select Final Fields
def select_final_fields(row):
    return row[1].cast("text") + row[2].cast("text")

# Write Final Data
def write_final_data(row):
    return row[1].cast("text") + row[2].cast("text")