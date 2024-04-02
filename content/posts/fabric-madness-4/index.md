---
title: "Fabric Madness: Experiments"
summary: "Machine Learning Systems usually require carefull tuning of its parts. Be it the features being used, the model, the model's hyperparameters, etc... Often, improving the performance of a system is done in an **experimental** way, trying out different configurations, and seeing which one works best."
date: 2024-03-29T11:37:43Z
draft: true
showAuthor: true
authors:
  - "martimchaves"
tags:
  - "Microsoft Fabric"
  - "MLFlow"
series: ["Fabric Madness"]
series_order: 4
---

## Introduction

Machine Learning (ML) Systems usually require carefull tuning of its parts. Be it the features being used, the model, the model's hyperparameters, etc... Often, improving the performance of a system is done in an **experimental** way, trying out different configurations, and seeing which one works best! That's why having a good setup to carry out these experiments is fundamental in the development of ML Systems.

Throughout this series, we've been discussing how Fabric has an **Experiment** tool. This tool is essentially a wrapper for MLFlow, with the added benefit of allowing for easy collaboration between anyone in the workspace. We don't have to worry about taking care of the infrastructure that a collaborative MLFlow environment would require! MLFlow is itself a great open-source platform - creating, running, analysing results, and drawing conclusions is made easy.

In this post, we aim to demonstrate how MLFlow can be used within Fabric. We'll be going over:
- How does MLFlow work?
- Creating and Setting Experiments
- Running Experiments and Logging Results
- Analysing Results

## How does MLFlow work?

MLFlow can be seen, in a very simplified way, as a database and a set of utility functions to interact with that database. This is where the information about the experiments will be neatly stored. In Fabric, the underlying part of MLFlow is taken care of.

There are two main organisational structures in MLFlow - **experiments** and **runs**. An experiment is a group of runs. A run, put simply, is an execution of a code snippet. Often, this code snippet is a training of a model. An experiment is ideal to group runs that are related.

For each run, information can be logged and attached to it - for example, metrics, hyperparameters, tags, artifacts (sort of like files, useful for special plots saved as images), and even models! Attaching models to runs is fantastic, especially when registering the model in (something that we'll go deeper into in the next post).

Runs can be filtered and compared. This allows us to understand which runs were more sucessful, and select the best performing run and use its setup (for example, in deployment).
Keep in mind, in Fabric, it seems that runs from different experiments can't be compared.

Knowing this, it's time to get practical!

## Creating and Setting Experiments

As mentioned in the first post of this series, using the UI to create an Experiment is straightforward - we have to select Experiment from the **+ New** button, and choose a name.

![Creating an Experiment using the UI. Shows mouse hovering experiment, with + New dropdown open](./images/exp/exp-1.png "Fig. 1 - Creating an Experiment using the UI")

Once that is done, to use that Experiment in a Notebook, this command has to be added:
```python
import mlflow

experiment_name = "[name of the experiment goes here]"

# Set the experiment
mlflow.set_experiment(experiment_name)
```

Alternatively, an Experiment can be created from a Notebook, which requires one extra command:
Alternatively, an Experiment can be created from a Notebook, which requires one extra command:
```python
import mlflow

experiment_name = "[name of the experiment goes here]"

# First create the experiment
mlflow.create_experiment(name=experiment_name)

# Then select it
mlflow.set_experiment(experiment_name)
```

Note that, if an Experiment with that name already exists, `create_experiment` will throw an error. In that case you might want to use the following code snippet, where first the existence of an Experiment with a given name is checked, and only if it doesn't exist is it created.
Note that, if an Experiment with that name already exists, `create_experiment` will throw an error. In that case you might want to use the following code snippet, where first the existence of an Experiment with a given name is checked, and only if it doesn't exist is it created.

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

When we set the experiment, runs will be saved to that experiment.

## Running Experiments and Logging Results

To signal a run that ought to be saved to the experiment we can use the following code snippet in a Notebook cell:

```python

# Start the training job with `start_run()`
with mlflow.start_run(run_name="example_run") as run:
    # rest of the code goes here

```

An example of a general workflow is one where a model is trained, and its score and hyper-parameters are logged. Here's an example of code that would do that:

When we set the experiment, runs will be saved to that experiment.

## Running Experiments and Logging Results

To signal a run that ought to be saved to the experiment we can use the following code snippet in a Notebook cell:

```python

# Start the training job with `start_run()`
with mlflow.start_run(run_name="example_run") as run:
    # rest of the code goes here

```

An example of a general workflow is one where a model is trained, and its score and hyper-parameters are logged. Here's an example of code that would do that:


```python
hyper_params = {"alpha": 0.5, "beta": 1.2}

# Start the training job with `start_run()`
with mlflow.start_run(run_name="simple_training") as run:
with mlflow.start_run(run_name="simple_training") as run:
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

## Analysing Results

## Analysing Results
