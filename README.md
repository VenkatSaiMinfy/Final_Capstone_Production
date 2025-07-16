sudo apt install postgresql postgresql-contrib

sudo -u postgres psql

sudo -u postgres psql -c "CREATE DATABASE lead_scoring_db;"

sudo -u postgres psql

\l

\c lead_scoring_db

select * from lead_data_uploaded limit 10;

pip install "apache-airflow==2.10.0" \
--constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.0/constraints-3.8.txt"


export AIRFLOW_HOME=$(pwd)/airflow


airflow db init

airflow users create \
    --username admin \
    --firstname Venkat \
    --lastname Sai \
    --role Admin \
    --email venkat@example.com


airflow webserver --port 8080
airflow scheduler

