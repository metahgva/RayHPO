import sys
import transformers
import argparse
import os
import ray
import mlflow
import numpy as np
import pandas as pd 
from ray import tune, air
from datasets import Dataset
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments, pipeline, TrainerCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ray.air import session
from ray.air.integrations.mlflow import MLflowLoggerCallback
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE" # Disabling hugging face MLFlow logger in favor of Ray MLFlow logger 

print("Initializing Ray Cluster...")
service_host = os.environ["RAY_HEAD_SERVICE_HOST"]
service_port = os.environ["RAY_HEAD_SERVICE_PORT"]
ray.init(f"ray://{service_host}:{service_port}")

class CustomCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Report metrics to Ray Tune
        session.report({"eval_accuracy": state.best_metric})

def load_data(file_name):
    # Load from CSV
    df = pd.read_csv(file_name).fillna("")
    
    # Encode labels
    df["label"] = df["label"].replace(["neutral","positive","negative"],[0,1,2]) 
    return df

def split(df):
    df_train, df_test, = train_test_split(df, stratify=df["label"], test_size=0.1, random_state=42)
    df_train, df_val = train_test_split(df_train, stratify=df_train["label"],test_size=0.1, random_state=42)
    print("Samples in train      : {:d}".format(df_train.shape[0]))
    print("Samples in validation : {:d}".format(df_val.shape[0]))
    print("Samples in test       : {:d}".format(df_test.shape[0]))
    
    return df_train, df_val, df_test


def prep_datasets(df_train, df_val, df_test, tokenizer):
    dataset_train = Dataset.from_pandas(df_train)
    dataset_val = Dataset.from_pandas(df_val)
    dataset_test = Dataset.from_pandas(df_test)

    dataset_train = dataset_train.map(lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length", max_length=315), batched=True)
    dataset_val = dataset_val.map(lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length", max_length=315), batched=True)
    dataset_test = dataset_test.map(lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length" , max_length=315), batched=True)

    dataset_train.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    dataset_val.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    dataset_test.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    
    return dataset_train, dataset_val, dataset_test


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy" : accuracy_score(predictions, labels)}

def train(config, epochs, data_dir):
    df = load_data(data_dir)
    df_train, df_val, df_test = split(df)
    
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels=3)
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    dataset_train, dataset_val, dataset_test = prep_datasets(df_train, df_val, df_test, tokenizer)

    args = TrainingArguments(
        output_dir="./outputs",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        adam_beta1=config["adam_beta1"],
        save_strategy="epoch",
        load_best_model_at_end=False,
        skip_memory_metrics=True,
        optim="adamw_torch",
        report_to="mlflow",
        metric_for_best_model="eval_accuracy",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics,
        callbacks=[CustomCallback()]
    )

    trainer.train()
    accuracy_test = trainer.predict(dataset_test).metrics["test_accuracy"]
    print("Test accuracy: {:.2f}".format(accuracy_test))
    trainer.save_model("./outputs/model")

def main():

    parser = argparse.ArgumentParser(description="Hyperparameter tuning a FinBERT model using the Sentiment Analysis for Financial News dataset.")
    parser.add_argument("--data", help="Path to CSV dataset.", required=False, default="/mnt/data/financial-news/data.csv", type=str)
    parser.add_argument("--epochs", help="Training epochs.", required=False, default=3, type=int)
    parser.add_argument("--trials", help="Number of trials.", required=False, default=4, type=int)
    args = parser.parse_args()

    config = {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "weight_decay": tune.choice([0, 0.1, 0.001]),
        "adam_beta1": tune.choice([0.9, 0.8])
    }

    train_fn = tune.with_parameters(train, epochs=args.epochs, data_dir=args.data)
    tuner = tune.Tuner(
        tune.with_resources(train_fn, resources={"cpu": 2, "gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="eval_accuracy",
            mode="max",
            num_samples=args.trials
        ),
        run_config=air.RunConfig(
            name="mlflow",
            callbacks=[MLflowLoggerCallback(experiment_name="finbert-hyperparameter-tuning", save_artifact=True)]
        ),
        param_space=config,
    )
    results = tuner.fit()
    print(results)
    
if __name__ == "__main__":
    main()
