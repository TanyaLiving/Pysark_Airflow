## ***MLFlow + Docker***

This project is one of the options for using MLFlow and docker to run experiments in an isolated docker environment.

Dataframe contains movie reviews, it was placed on Google Drive.

All necessary parameters are in a file [config.yaml](https://github.com/TanyaLiving/MLFlow/blob/master/config/config.yaml).

2 containers have been created for training the model and tracking experiments on the local MLflow server.

Containers are linked via [docker-compose.yaml](https://github.com/TanyaLiving/MLFlow/blob/master/docker-compose.yaml).
