---
layout: post
title: MLflow hosted models
date: 2024-06-15
description: Notes on MLflow hosted models
---

## 1. Model Training and Logging

Train your custom model and log it to MLflow using `mlflow.log_model()`. You can specify the model's signature and other metadata like:


```python
with mlflow.start_run():
    # training logic here
    
    mlflow.pyfunc.log_model("model_name", 
                            python_model=model, 
                            signature=model_signature)
```

## 2. Model Saving

MLflow supports saving models in a variety of formats. For custom models, you might want to use the `pyfunc` flavor, which allows you to save Python functions or classes as models.

## 3. Model Serving

You can serve your model locally with:

`mlflow models serve -m <run_id>/model -p <port>`

Where `<run_id>` is the ID from your MLflow run where you logged the model.

For a more production-like setup, you can deploy models using MLflow's model server, which can be set up on various platforms (e.g., Kubernetes, AWS SageMaker, Azure ML, etc.).

## 4. Custom Pyfunc Models

If your model doesn't fit into standard frameworks, you can create a custom PythonModel class:


```python
from mlflow.pyfunc import PythonModel

class CustomModel(PythonModel):
    def __init__(self):
        # init here

    def predict(self, context, model_input):
        # inference logic here
        return model_output

mlflow.pyfunc.log_model("model_name", python_model=CustomModel())
```

## 5. Deployment

Once logged, you can deploy your model for serving via REST API, for instance, or integrate it into your application using the `mlflow.pyfunc.load_model()` function to load the model in your code.

This can also be called from the command line with

```
mlflow models serve -m models:/MODEL_NAME/VERSION -p 5005 -h 0.0.0.0 --no-conda
```

This may require setting the default URI as an environment variable.

```
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

## 6. Containerization

MLflow also supports packaging models into Docker containers, which can be very useful for deployment consistency across environments.

Remember, when hosting custom models, ensuring that all dependencies are correctly managed is crucial since MLflow will use the environment where the model was logged to serve the model. You might need to specify a `conda.yaml` or `requirements.txt` file with your dependencies when logging the model.

This gives you a broad outline. The exact steps might vary based on your specific use case, the complexity of your model, and the deployment environment.


