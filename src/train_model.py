from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.naive_bayes import MultinomialNB
import mlflow
import yaml
import hyperopt
from hyperopt import hp
from hyperopt import space_eval, STATUS_OK, fmin, tpe
from functions import (
    model_result,
    conf_matrix,
    plot_conf_matrix,
    feature_importance,
    r_a_score,
)

from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer

spark = SparkSession.builder()

config = yaml.safe_load(open("./config/config.yaml"))["train_model"]

## Preprocessed data

test_preprocessed = pd.read_csv(config["data_path_test"], sep=";")

train_preprocessed = pd.read_csv(config["data_path_train"], sep=";")

train_X = train_preprocessed.review.array
train_y = train_preprocessed.sentiment.array

test_X = test_preprocessed.review.array
test_y = test_preprocessed.sentiment.array

class_names = ["negative", "positive"]

mlflow.set_tracking_uri("http://mlflow_web_servise:5000/")
exp_id = mlflow.set_experiment(experiment_name=config["experiment_name"])


# Preparation
RANDOM_STATE = 42
cv = StratifiedShuffleSplit(
    n_splits=config["n_splits"],
    test_size=config["test_size"],
    random_state=RANDOM_STATE,
)

# Search
tokenizer = Tokenizer(inputCol="review", outputCol="tokens")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression()
MB = NaiveBayes()

if config["model"] == "log_reg":
    my_pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

elif config["model"] == "MultinomialNB":
    my_pipeline = Pipeline(stages=[tokenizer, hashingTF, MB])

with mlflow.start_run(
    run_name=f'PARENT_{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
):

    search_space = hp.choice(
        "classifier_type",
        [
            {
                "type": "naive_bayes",
                "clf__alpha": hp.uniform(
                    "clf__alpha",
                    low=config["low_alpha_NB"],
                    high=config["high_alpha_NB"],
                ),
            },
            {
                "type": "log_reg",
                "clf__penalty": hp.choice(
                    "clf__penalty", config["clf__penalty_choice"]
                ),
                "clf__C": hp.uniform(
                    label="clf__C", low=config["low_C"], high=config["high_C"]
                ),
            },
        ],
    )

    def objective(x_var):

        with mlflow.start_run(
            run_name=f'CHILD_{datetime.now().strftime("%H:%M:%S")}',
            experiment_id=exp_id.experiment_id,
            nested=True,
        ) as child_run:
            classifier_type = config["model"]
            if classifier_type == "log_reg":
                estimator = my_pipeline.set_params(clf__solver=config["solver"]).fit(
                    train_X, train_y
                )
            elif classifier_type == "naive_bayes":
                estimator = my_pipeline.fit(train_X, train_y)
            score = cross_val_score(
                estimator=estimator,
                X=train_X,
                y=train_y,
                scoring=config["scoring"],
                cv=config["cross_val_score_cv"],
                error_score="raise",
            ).mean()
            mlflow.log_params(x_var)
            print(estimator)
            mlflow.sklearn.log_model(estimator, "model")
            print('mlflow.sklearn.log_model(estimator, "model")')

            mlflow.log_metric("ROC_AUC_TRAIN", score)
            print(x_var)

            return {"loss": 1 - score.mean(), "params": x_var, "status": STATUS_OK}

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=config["max_evals"],
        rstate=np.random.default_rng(RANDOM_STATE),
        show_progressbar=True,
    )

    best_params = space_eval(search_space, best)
    print(best_params)

    # Best model

    if best_params["type"] == "log_reg":
        my_pipeline_tuned = Pipeline(
            [
                ("vectorizer", TfidfVectorizer()),
                (
                    "clf",
                    LogisticRegression(
                        C=best_params["clf__C"],
                        penalty=best_params["clf__penalty"],
                        solver=config["solver"],
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    elif best_params["type"] == "naive_bayes":
        my_pipeline_tuned = Pipeline(
            [
                ("vectorizer", TfidfVectorizer()),
                ("clf", MultinomialNB(alpha=best_params["clf__alpha"])),
            ]
        )

    best_result = model_result(
        my_pipeline_tuned,
        train_X,
        train_y,
        test_X,
    )
    mlflow.sklearn.log_model(my_pipeline_tuned, "model")

    ROC_AUC_test_LR_tuned = round(
        roc_auc_score(test_y, my_pipeline_tuned.predict_proba(test_X)[:, 1]), 2
    )
    mlflow.log_metric("ROC_AUC_TEST", ROC_AUC_test_LR_tuned)

    mlflow.log_params(best_params)

    conf_matr = conf_matrix(test_y, class_names, best_result)
    plot_conf_matrix(config["experiment_name"], class_names, conf_matr)
    r_a_score(my_pipeline_tuned, train_X, train_y, test_X, test_y)

    # Feature importance plots
    if best_params["type"] == "log_reg":
        feature_importance(my_pipeline_tuned, config["fi_plot_n_features"])
