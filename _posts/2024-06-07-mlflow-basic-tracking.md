---
layout: post
title: MLflow basic tracking
date: 2024-06-07
description: Basic operations and experiment tracking with MLflow 
---

## Using MLflow for Experiment Tracking and Metric Logging

### Basic MLflow Usage

#### Starting an MLflow Run
An MLflow run is the basic unit of execution for logging parameters, metrics, and artifacts. To start a run:

```python
import mlflow

with mlflow.start_run():
    # your code here
```

#### Auto Logging
MLflow supports automatic logging for libraries like scikit-learn, TensorFlow, and PyTorch. Auto logging captures metrics, parameters, and models without additional manual logging code.

Example with scikit-learn:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

mlflow.sklearn.autolog()

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Metrics and parameters are automatically logged
```

#### Manual Tracking
For greater control, you can manually log parameters, metrics, and artifacts. This is particularly useful for unsupported libraries or custom workflows.

##### Logging Parameters
```python
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
```

##### Logging Metrics
```python
with mlflow.start_run():
    accuracy = 0.95
    mlflow.log_metric("accuracy", accuracy)

    # Log multiple metrics at different steps
    for i in range(10):
        mlflow.log_metric("loss", i * 0.1, step=i)
```

##### Logging Artifacts
Artifacts can include plots, datasets, or any files relevant to your experiment:

```python
import matplotlib.pyplot as plt

with mlflow.start_run():
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig("plot.png")
    mlflow.log_artifact("plot.png")
```

##### Logging Models
You can log models for reproducibility or deployment:

```python
from sklearn import svm

with mlflow.start_run():
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    mlflow.sklearn.log_model(clf, "model")
```

### Managing Experiments
Experiments are organizational units in MLflow that group related runs. By default, runs are logged under the "Default" experiment, but you can specify a custom experiment:

```python
mlflow.set_experiment("My Experiment")
with mlflow.start_run():
    # Experiment code here
```

### Viewing Results
To explore your experiments, start the MLflow UI:

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to see all experiments, parameters, metrics, and artifacts.

### Complete Example
Here’s a full script combining auto and manual logging:

```python
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

mlflow.set_experiment("Iris Experiment")
mlflow.sklearn.autolog()

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=100, max_depth=6)
    rf.fit(X_train, y_train)

    # Manual logging
    mlflow.log_param("random_state", 42)
    score = rf.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", score)

    # Plot feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    plt.figure()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [f'feature {i}' for i in indices])
    plt.savefig("importance.png")
    mlflow.log_artifact("importance.png")

    # Log the model
    mlflow.sklearn.log_model(rf, "model")
```

### LLMs

MLflow also provides capabilities for evaluating language models (LLMs). These include heuristic metrics and novel approaches like "LLM-as-a-Judge" metrics.

#### Heuristic Metrics
Traditional metrics for text comparison:

```python
from mlflow.evaluate import evaluate

data = {
    "inputs": ["What is MLflow?", "What is Spark?"],
    "ground_truth": [
        "MLflow manages and tracks the ML lifecycle.", 
        "Spark is for big data processing."
    ],
}

def model_predict(inputs):
    return ["MLflow helps with ML lifecycle.", "Spark processes big data."]

results = evaluate(
    model=model_predict,
    data=data,
    targets="ground_truth",
    metrics=["token_count", "toxicity", "bleu", "rougeL"]
)

print(results.metrics)
```

#### LLM-as-a-Judge Metrics
Use LLMs to evaluate other LLMs:

```python
from mlflow.metrics.genai import answer_correctness, answer_relevance
from mlflow.evaluate import evaluate

data = {
    "inputs": ["What is MLflow?", "What is Spark?"],
    "ground_truth": ["MLflow manages the ML lifecycle.", "Spark is for big data processing."],
}

def model_predict(inputs):
    return ["MLflow aids in ML management.", "Spark deals with large datasets."]

metrics = [answer_correctness(), answer_relevance()]

results = evaluate(
    model=model_predict,
    data=data,
    targets="ground_truth",
    extra_metrics=metrics
)

print(results.metrics)
```

#### Custom Metrics
Define a custom metric, such as professionalism:

```python
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

professionalism_metric = make_genai_metric(
    name="professionalism",
    definition="Evaluate the professionalism of the response.",
    grading_prompt="Rate on a scale of 1-5 where 1 is unprofessional and 5 is very professional.",
    examples=[
        EvaluationExample(
            input="Explain MLflow.", 
            output="MLflow is for ML lifecycle management.", 
            score=5, 
            justification="Formal and informative."
        ),
        EvaluationExample(
            input="Explain MLflow.", 
            output="MLflow? It's just for ML stuff.", 
            score=2, 
            justification="Informal, lacks detail."
        )
    ]
)

results = evaluate(
    model=model_predict,
    data=data,
    targets="ground_truth",
    extra_metrics=[professionalism_metric]
)

print(results.metrics)
```

