from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.email import EmailOperator
from airflow.providers.telegram.operators.telegram import TelegramOperator
from airflow.operators.python_operator import PythonOperator
import requests
from airflow.utils.dates import days_ago
from airflow.models import Variable

site = Variable.get("site")

dag = DAG('apartment_tracking', 
        description='apartment tracking', 
        # start_date = pendulum.now(),
        schedule_interval='*/5 * * * *',
        start_date = days_ago(1),
        catchup=False,
)

def flat():
        r = requests.get(site)
        return r.text.find('ОШИБКА 404. ТАКОЙ СТРАНИЦЫ НЕ СУЩЕСТВУЕТ') != -1
        

get_data = PythonOperator(task_id='python_task', python_callable=flat, dag=dag)

send_telegram_message = TelegramOperator(
        task_id='send_telegram_message',
        telegram_conn_id='telegram_default',
        chat_id='-805367692',
        text="Flat's available",
        dag=dag,
)

get_data >> send_telegram_message