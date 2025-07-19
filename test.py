import psycopg2

# Define connection parameters
REDSHIFT_HOST = "capstoneworkgroup.731239205085.ap-south-1.redshift-serverless.amazonaws.com"
REDSHIFT_PORT = 5439  # default port
REDSHIFT_DB = "dev"
REDSHIFT_USER = "admin"
REDSHIFT_PASS = "Venky643!!"

S3_FILE_1 = 's3://capstonedataminfy/preprocessed_train_data.csv'
S3_FILE_2 = 's3://capstonedataminfy/user_uploaded_preprocessed.csv'

# Target Redshift tables
REDSHIFT_TABLE_1 = 'preprocessed_train_data'
REDSHIFT_TABLE_2 = 'user_uploaded_preprocessed'

# Connect to Redshift
conn = psycopg2.connect(
    host=REDSHIFT_HOST,
    port=REDSHIFT_PORT,
    dbname=REDSHIFT_DB,
    user=REDSHIFT_USER,
    password=REDSHIFT_PASS
)
cur = conn.cursor()

try:
    # COPY commands using IAM_ROLE default
    copy_sql_1 = f"""
        COPY {REDSHIFT_TABLE_1}
        FROM '{S3_FILE_1}'
        IAM_ROLE default
        FORMAT AS CSV
        IGNOREHEADER 1;
    """
    copy_sql_2 = f"""
        COPY {REDSHIFT_TABLE_2}
        FROM '{S3_FILE_2}'
        IAM_ROLE default
        FORMAT AS CSV
        IGNOREHEADER 1;
    """

    print('Loading first file...')
    cur.execute(copy_sql_1)
    print('First file loaded.')

    print('Loading second file...')
    cur.execute(copy_sql_2)
    print('Second file loaded.')

    conn.commit()

except Exception as e:
    print(f"Error loading data: {e}")
    conn.rollback()
finally:
    cur.close()
    conn.close()
