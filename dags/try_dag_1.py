from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.email import EmailOperator
from airflow.providers.telegram.operators.telegram import TelegramOperator

dag = DAG('Classification_pipe', 
description='print example text', 
start_date = pendulum.now()
)


docker_prearation = DockerOperator(
    task_id = 'docker_prearation',
    image = 'image.preparation',
    container_name = 'preparation',
    auto_remove = 'force',
    working_dir = '/Pyspark_Airflow_Docker_Compose/',
    command = [ "python3",  "./src/preparation.py"],
    dag = dag
)

docker_train_model = DockerOperator(
    task_id = 'docker_train_model',
    image = 'image.preparation',
    container_name = 'train_model',
    auto_remove = 'force',
    working_dir = '/Pyspark_Airflow_Docker_Compose/',
    command = ["python3", "./src/train_model.py"],
    network_mode  = 'pyspark_airflow_docker_compose_default',
    dag = dag
)

send_telegram_message = TelegramOperator(
        task_id='send_telegram_message',
        telegram_conn_id='telegram_default',
        chat_id='-805367692',
        text='hey',
        dag=dag,
        # trigger_rule='one_failed'
)

docker_prearation >> docker_train_model
docker_train_model >> send_telegram_message