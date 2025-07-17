sudo apt install postgresql postgresql-contrib

sudo -u postgres psql

sudo -u postgres psql -c "CREATE DATABASE lead_scoring_db;"

sudo -u postgres psql

\l

\c lead_scoring_db

select * from lead_data_uploaded limit 10;

pip install "apache-airflow==2.10.0" \
--constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.0/constraints-3.8.txt"


🪄 Step 1: Open your shell config file
The shell you use in WSL is typically bash, so edit your .bashrc:

bash
Copy
Edit
nano ~/.bashrc
Or if you're using ZSH, use:

bash
Copy
Edit
nano ~/.zshrc
🧩 Step 2: Add the following line at the bottom
Add this line:

bash
Copy
Edit
export AIRFLOW_HOME=~/Final_Capstone_Project/lead_scoring_project/src/airflow
💡 If you want to make sure your Airflow environment always activates too, you can optionally add:

bash
Copy
Edit
conda activate lead_scoring_system
Your .bashrc could look like this at the end:

bash
Copy
Edit
# Airflow project config
export AIRFLOW_HOME=~/Final_Capstone_Project/lead_scoring_project/src/airflow
conda activate lead_scoring_system
✅ Step 3: Save and exit
In nano:

Press Ctrl + O → hit Enter to save.

Press Ctrl + X to exit.

🔁 Step 4: Reload your shell
To apply changes immediately, run:

bash
Copy
Edit
source ~/.bashrc
Now, every time you open a new WSL terminal, these will happen automatically:

Your AIRFLOW_HOME is set correctly.

Airflow uses your src/airflow folder for all its needs (like airflow.cfg, airflow.db, logs, etc.).

If you added the line, your Anaconda env (lead_scoring_system) will also activate automatically.

🧪 Step 5: Verify
To confirm:

bash
Copy
Edit
echo $AIRFLOW_HOME
You should see:

swift
Copy
Edit
/home/venkat/Final_Capstone_Project/lead_scoring_project/src/airflow
Also, run:

bash
Copy
Edit
which airflow
It should point to your conda environment’s bin directory.

🧼 Tip: Remove old Airflow folders
To prevent confusion, remove default ~/airflow folder if it exists:

bash
Copy
Edit
rm -rf ~/airflow
✅ Final Notes
From now on:

You can run airflow webserver and airflow scheduler from anywhere, and it’ll still use src/airflow.

All files like airflow.cfg, logs/, airflow.db, etc. will be inside your project — clean and modular.

airflow db init

airflow users create \
    --username admin \
    --firstname Venkat \
    --lastname Sai \
    --role Admin \
    --email venkat@example.com


airflow webserver --port 8080
airflow scheduler



CREATE TABLE public.lead_data (
    "Prospect ID" text,
    "Lead Number" bigint,
    "Lead Origin" text,
    "Lead Source" text,
    "Do Not Email" text,
    "Do Not Call" text,
    "Converted" bigint,
    "TotalVisits" double precision,
    "Total Time Spent on Website" bigint,
    "Page Views Per Visit" double precision,
    "Last Activity" text,
    "Country" text,
    "Specialization" text,
    "How did you hear about X Education" text,
    "What is your current occupation" text,
    "What matters most to you in choosing a course" text,
    "Search" text,
    "Magazine" text,
    "Newspaper Article" text,
    "X Education Forums" text,
    "Newspaper" text,
    "Digital Advertisement" text,
    "Through Recommendations" text,
    "Receive More Updates About Our Courses" text,
    "Tags" text,
    "Lead Quality" text,
    "Update me on Supply Chain Content" text,
    "Get updates on DM Content" text,
    "Lead Profile" text,
    "City" text,
    "Asymmetrique Activity Index" text,
    "Asymmetrique Profile Index" text,
    "Asymmetrique Activity Score" double precision,
    "Asymmetrique Profile Score" double precision,
    "I agree to pay the amount through cheque" text,
    "A free copy of Mastering The Interview" text,
    "Last Notable Activity" text
);



import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue import DynamicFrame

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Script generated for node Amazon S3
AmazonS3_node1752777278258 = glueContext.create_dynamic_frame.from_catalog(database="capstone database", table_name="lead_scoring_csv", transformation_ctx="AmazonS3_node1752777278258")

# Script generated for node Amazon Redshift
AmazonRedshift_node1752777280887 = glueContext.write_dynamic_frame.from_options(
    frame=AmazonS3_node1752777278258,
    connection_type="redshift",
    connection_options={
        "redshiftTmpDir": "s3://aws-glue-assets-122610482767-ap-south-1/temporary/",
        "useConnectionProperties": "true",
        "dbtable": "public.lead_data",
        "connectionName": "Redshift connection",
        "preactions": """CREATE TABLE IF NOT EXISTS public.lead_data (
            "Prospect ID" VARCHAR,
            "Lead Number" BIGINT,
            "Lead Origin" VARCHAR,
            "Lead Source" VARCHAR,
            "Do Not Email" VARCHAR,
            "Do Not Call" VARCHAR,
            "Converted" BIGINT,
            "TotalVisits" BIGINT,
            "Total Time Spent on Website" BIGINT,
            "Page Views Per Visit" DOUBLE PRECISION,
            "Last Activity" VARCHAR,
            "Country" VARCHAR,
            "Specialization" VARCHAR,
            "How did you hear about X Education" VARCHAR,
            "What is your current occupation" VARCHAR,
            "What matters most to you in choosing a course" VARCHAR,
            "Search" VARCHAR,
            "Magazine" VARCHAR,
            "Newspaper Article" VARCHAR,
            "X Education Forums" VARCHAR,
            "Newspaper" VARCHAR,
            "Digital Advertisement" VARCHAR,
            "Through Recommendations" VARCHAR,
            "Receive More Updates About Our Courses" VARCHAR,
            "Tags" VARCHAR,
            "Lead Quality" VARCHAR,
            "Update me on Supply Chain Content" VARCHAR,
            "Get updates on DM Content" VARCHAR,
            "Lead Profile" VARCHAR,
            "City" VARCHAR,
            "Asymmetrique Activity Index" VARCHAR,
            "Asymmetrique Profile Index" VARCHAR,
            "Asymmetrique Activity Score" BIGINT,
            "Asymmetrique Profile Score" BIGINT,
            "I agree to pay the amount through cheque" VARCHAR,
            "A free copy of Mastering The Interview" VARCHAR,
            "Last Notable Activity" VARCHAR
        );"""
    },
    transformation_ctx="AmazonRedshift_node1752777280887"
)

job.commit()