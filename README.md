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

