import snowflake.connector
import os
from dotenv import load_dotenv

load_dotenv()

conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)

cur = conn.cursor()
sql = 'SELECT ATMOSPHERIC_TEMP, HUMIDITY, SOIL_TEMP, SOIL_MOISTURE, PREVIOUS_SOIL_MOISTURE, FUTURE_SOIL_MOISTURE,RAINFALL FROM staging.stg_data'

df = cur.execute(sql).fetch_pandas_all()

