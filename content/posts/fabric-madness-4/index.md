---
title: "Fabric Madness: Experiments"
summary: "In this series of posts titled Fabric Madness, we're going to be diving deep into some of the most interesting features of Microsoft Fabric, for an end-to-end demonstration of how to train and use a machine learning model."
date: 2024-03-29T11:37:43Z
draft: true
showAuthor: true
authors:
  - "martimchaves"
  - "rogernoble"
tags:
  - "Microsoft Fabric"
  - "MLFlow"
series: ["Fabric Madness"]
series_order: 4
---

## Introduction

Once that is done, to use that Experiment in a Notebook, this command has to be added:
```python
import mlflow

experiment_name = "[name of the experiment goes here]"

# Set the experiment
mlflow.set_experiment(experiment_name)
```

Alternatively, an Experiment can be created from the Notebook, which requires one extra command:
```python
import mlflow

experiment_name = "[name of the experiment goes here]"

# First create the experiment
mlflow.create_experiment(name=experiment_name)

# Then select it
mlflow.set_experiment(experiment_name)
```

Note that, if an Experiment with that name already exists, `create_experiment` will throw an error. In that case you might want to use the following code snippet, where first the existence of an Experiment with a given name is checked, and only if it doesn't exist is it create.

```python
import mlflow

experiment_name = "[name of the experiment goes here]"

# Check if experiment exists
# if not, create it
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(name=experiment_name)

# Set experiment
mlflow.set_experiment(experiment_name)
```

When we set the experiment, runs will be saved to that experiment. To signal that a code snippet is a run that ought to be saved to the experiment we can use the following code snippet in a Notebook cell:

```python
hyper_params = {"alpha": 0.5, "beta": 1.2}

# Start the training job with `start_run()`
with mlflow.start_run() as run:
	# Create model and dataset
	model = create_model(hyper_params)
	X, y = create_dataset()
	
	# Train model
	model.fit(X, y)

	# Calculate score
	score = lr.score(X, y)

	# Log metrics and hyper-parameters
	print("Log metric.")
	mlflow.log_metric("score", score)

	print("Log params.")
	mlflow.log_param("alpha", hyper_params["alpha"])
	mlflow.log_param("beta", hyper_params["beta"])
		
```

The general workflow is described in the code snippet above. After setting the experiment and starting a run, a model is trained, and its score and hyper-parameters are logged.

Another very useful tool in Fabric that should be introduced now is the ML Model tool. This tool is essentially a wrapper for the MLFlow Model Registry. It allows us to register models and keep track of different versions and their respective performances. For this case study, this was perfect. Each of the three different models were registered under a different name, and each version was saved, along with its score. To do that, a couple of extra lines are needed:


```python
hyper_params = {"alpha": 0.5, "beta": 1.2}

# Start the training job with `start_run()`
with mlflow.start_run() as run:
	... (previous code)
	
	# Log a model
	mlflow.tensorflow.log_model(lr, "my_model_1")
	
	# Get model URI
	model_uri = f"runs:/{run.info.run_id}/my_model_1"
	
	# Select Model Name
	model_name = "Model1"
	
	# Register Model
	result = mlflow.register_model(model_uri, model_name)
```

In this case, if a ML Model with the `model_name` already exists, a new version is added. If it doesn't exist, an ML Model is created with that name and the logged model is considered the first version.

An ML Model can also be created via Fabric's UI. Model versions of said ML Model can be imported from runs from several different Experiments.

![Creating a ML Model using the UI. Shows mouse hovering experiment, with + New dropdown open](./images/exp-2.png "Fig. 1 - Creating a ML Model using the UI")

Considering this case study, an Experiment was created for each of the three models. Several runs were executed, testing different values for the learning rate, and registering a new version of each model along the way.


```python

model_dict = {
    'model_s': create_model_1,  # small
    'model_m': create_model_2,  # medium
    'model_l': create_model_3   # large

}

input_shape = X_train_scaled_df.shape[1]
epochs = 100

for model_name in model_dict:
    
    # create mlflow experiment
    experiment_name = "3experiment_" + model_name

    # Check if experiment exists
    # if not, create it
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name)

    # Set experiment
    mlflow.set_experiment(experiment_name)

    learning_rate_list = [0.001, 0.01, 0.1]

    for lr in learning_rate_list:

        with mlflow.start_run() as run:
            # Create model and dataset
            model = model_dict[model_name](input_shape)

            # Train model
            history, best_model = compile_and_train(model,
                                                    X_train_scaled_df, y_train,
                                                    X_validation_scaled_df, y_validation,
                                                    epochs,
                                                    lr)
            
            
            # Calculate score
            brier_score = evaluate_model(best_model, X_test_scaled_df, y_test)

            # Log metrics and hyper-parameters
            mlflow.log_metric("brier", brier_score)

            # Log hyper-param
            mlflow.log_param("lr", lr)

            print("Logging model!")

            # Log model
            predictions = best_model.predict(X_train_scaled_df)

            # Signature required for model loading later on
            signature = infer_signature(np.array(X_train_scaled_df), predictions)
            model_file_name = model_name + "_file"
            mlflow.tensorflow.log_model(best_model, model_file_name, signature=signature)
            
            # Get model URI
            model_uri = f"runs:/{run.info.run_id}/{model_file_name}"
            
            # Register Model
            result = mlflow.register_model(model_uri, model_name)

```

After that was done, the next step was selecting the best model. This could have been done visually, using the UI, opening each experiment, selecting the List View, and selecting all of the available runs.

![Experiment window](./images/exp-3.png "Fig. 2 - Inspecting Experiment")

Alternatively, it can also be done via code, by getting all of the versions of all of the ML Models performance, and selecting the version with the best score.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

mlmodel_names = list(model_dict.keys())
best_score = 2
metric_name = "brier"
best_model = {"model_name": "", "model_version": -1}

for mlmodel in mlmodel_names:

	model_versions = client.search_model_versions(filter_string=f"name = '{mlmodel}'")

	for version in model_versions:

		# Get metric history for Brier score and run ID
		metric_history = client.get_metric_history(run_id=version.run_id,
		                                           key=metric_name)

		# If score better than best score, save model name and version
		if metric_history:
			last_value = metric_history[-1].value
			if last_value < best_score:
				best_model["model_name"] = mlmodel
				best_model["model_version"] = version
				best_score = last_value
		else:
			continue
```

After finding the best model, using it to get the final predictions can be be done using the following code snippet:

```python
loaded_best_model = mlflow.pyfunc.load_model(f"models:/{best_model['model_name']}/{best_model['model_version'].version}")
final_brier_score = evaluate_model(loaded_best_model, X_test_scaled_df, y_test)
print(f"Best final Brier score: {final_brier_score}")
```