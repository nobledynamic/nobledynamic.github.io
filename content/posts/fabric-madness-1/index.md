---
title: "Fabric Madness: predicting basketball games with Microsoft Fabric"
summary: "In this series of posts titled Fabric Madness, we're going to be diving deep into some of the most interesting features of Microsoft Fabric, for an end-to-end demonstration of how to train and use a machine learning model."
date: 2024-03-26T11:37:43Z
draft: true
showAuthor: true
authors:
  - "martimchaves"
  - "rogernoble"
tags:
  - "Microsoft Fabric"
series: ["Fabric Madness"]
series_order: 1
---

[//]: # "Mostly using active voice (we did...) instead of passive voice (x was done...) to sound more direct and engaging. Passive voice used when we want to give emphasis to the object (A Lakehouse was created - emphasis on the Lakehouse)"

## Introduction

At the time of writing, it's basketball season in the United States, and there is a lot of excitement around the men's and women's college basketball tournaments. The format is single elimination, so over the course of several rounds, teams are eliminated, till eventually we get a champion. This tournament is not only a showcase of upcoming basketball talent, but, more importantly, a fertile ground for data enthusiasts like us to analyze trends and predict outcomes.

One of the great things about sports (perhaps the only thing!) is that there is lots of data available, and we at [Noble Dynamic](https://nobledynamic.com/) wanted to take a crack at it :nerd_face:.

In this series of posts titled *Fabric Madness*, we're going to be diving deep into some of the most interesting features of [Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric), for an end-to-end demonstration of how to train and use a machine learning model.

In this first blog post, we'll be going over:
- A frist look of the data using [Data Wrangler](https://learn.microsoft.com/en-us/fabric/data-science/data-wrangler).
- Exploratory Data Analysis (EDA) and Feature Engineering with the help of PySpark
- Tracking the performance of different Machine Learning (ML) Models using Experiments
- Selecting the best performing model using the ML Model functionality
- Predicting the winner of this year's tournament

Let's get to the first step, getting and processing data to create a dataset with relevant features.

## The Data

The data used was obtained from the on-going Kaggle competition. That competition can be found [here](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/overview).

Among all of the interesting data available, our focus for this case study was on the match-by-match statistics. This data was available for both the regular seasons and the tournaments, going all the way back to 2003. For each match, besides the date, the teams that were playing, and their scores, other relevant features were made available, such as field goals made and personal fouls by each team.

### Loading the Data

The first step was creating a Fabric workspace. In this workspace is where all of the data and tools would be used.

After downloading all of the CSV files available, a **Lakehouse** was created. A Lakehouse, in simple terms, is storage for both Tables (structured) and Files (unstructured) data. The data in the created Lakehouse is available for every tool in the workspace.

Uploading the files was done using the UI:

![Uploading Files](./images/lakehouse/lakehouse3.png "Fig. 1 - Uploading Files")

Now that there was a Lakehouse with the CSV files, it was time to dig in, and get a first look at the data. To do that, we created a Notebook, using the UI, and attached the previously created lakehouse.

![Adding Lakehouse to Notebook](./images/lakehouse/lakehouse4.png "Fig. 2 - Adding Lakehouse to Notebook")

### First Look

After a quick data wrangling, it was found that, as expected with data from Kaggle, the quality was great. With no duplicates or missing values.

To do this, we used [Data Wrangler](https://learn.microsoft.com/en-us/fabric/data-science/data-wrangler), a tool built into Microsoft Fabric notebooks. Once an initial DataFrame has been created (Spark or pandas supported), Data Wrangler becomes available to use and can attach to any DataFrame in the Notebook. What's great is that it allows for easy analysis of loaded DataFrames. In a Notebook, after reading the files into PySpark DataFrames, in the "Data" section, the "Transform DataFrame in Data Wrangler" was selected,  and from there the several DataFrames were explored. Specific DataFrames can be chosen, carrying out a careful inspection.

![Clicking on data wrangler in MS Fabric](./images/data-wrangler/data-wrangler-1.png "Fig. 3 - Opening Data Wrangler")

![Analysing the DataFrame with Data Wrangler](./images/data-wrangler/data-wrangler-2.png "Fig. 4 - Analysing the DataFrame with Data Wrangler")

In the center, we have access to all of the rows of the loaded DataFrame. On the right, a **Summary** tab, showing that indeed there are no duplicates or missing values. Clicking in a certain column, summary statistics of that column will be shown.

On the left, in the **Operations** tab, there are several pre-built operations that can be applied to the DataFrame.

If it were the case that some DataFrames needed cleaning, those steps could also be done using the Data Wrangler, in a low-code format. The desired operation would have to be selected and applied. In this case, the data was already in good shape, so we moved on to EDA.

### EDA

A short Exploratory Data Analysis (EDA) followed, with the goal of getting a general idea of the data. Charts were plotted to get a sense of the distribution of the data and if there were any statistics that could be problematic due to, for example, very long tails.

![Example of an histogram of fields goal made. It shows a normal distribution](./images/eda/eda-1.png "Fig. 5 - Histogram of Fields Goal Made")

At a quick glance, it was found that the data available from the regular season had normal distributions, suitable to use in the creation of features. Knowing the importance that good features have in creating solid predictive systems, the next sensible step was to carry out feature engineering to extract relevant information from the data.

The goal was to create a dataset where each sample's input would be a set of features for a game, containing information of both teams. For example, both teams average field goals made for the regular season. The target for each sample, the desired output, would be 1 if Team 1 won the game, or 0 if Team 2 won the game (which was done by subtracting the scores). Here's a representation of the dataset:

| Team1ID | Team2ID | Team1Feat1 | Team2Feat2 | T1Score-T2Score | Target |
|:-------:|:-------:|:----------:|:----------:|:---------------:|:------:|
| 1       | 2       | 0.5        | 0.6        | 8               | 1      |
| 3       | 1       | 0.2        | 0.7        | 12              | 1      |
| 2       | 4       | 0.8        | 0.6        | -3              | 0      |

### Feature Engineering

The first feature that we decided to explore was win rate. Not only would it be an interesting feature to explore, but it would also provide a baseline score. This initial approach employed a simple rule: the team with the higher win rate would be predicted as the winner. This method provides a fundamental baseline against which the performance of more sophisticated predictive systems can be compared to.

To evaluate the accuracy of our predictions across different models, we adopted the Brier score. The Brier score is the mean of the square of the difference between the predicted probability (p) and the actual outcome (o) for each sample, and can be described by the following formula:

{{< katex >}}
\\(\Large Brier Score = \frac{1}{N} \sum_{i=1}^{N} (p_i - o_i)^2 \\)

Calculating this score is important, as it is used to quantify the accuracy of predictions, similar to how the Mean Squared Error works. However, this metric is especially useful for binary classification. The predicted probability will vary between 0 and 1, and the actual outcome will either be 0 or 1. Thus the Brier score will always be between 0 and 1. As we want the predicted probability to be as close to the actual outcome as possible, the lower the Brier score, the better, with 0 being the perfect score, and 1 the worst.

For the baseline, the previously mentioned dataset structure was followed. Each sample of the dataset was a match, containing information for Team 1 and Team 2, the teams that played in that match - specifically, their win rates for the regular season. The actual outcome was considered 1 if Team 1 won, or 0 if Team 2 won. To simulate a probability, the prediction was a normalised difference between T1's win rate and T2's win rate. For the maximum value of the difference between the win rates, the prediction would be 1. For the minimum value, the prediction would be 0. The Score Difference as the target will be more useful later on, when more features are available, and more complex models will be used.

```python
# Add the "outcome" column: 1 if T1_Score > T2_Score, else 0
tourney_df = tourney_df.withColumn("outcome", F.when(F.col("T1_Score") > F.col("T2_Score"), 1).otherwise(0))

# Adjust range from [-1, 1] to [0, 1]. If below .5 T1 loses, if above .5 T1 wins. If same win rate, assumed "draw" 
tourney_df = tourney_df.withColumn("probability", (F.col("T1_win_ratio") - F.col("T2_win_ratio") + 1) / 2)
```

After calculating the win rate, and then using it to predict the outcomes, we got a Brier score of **0.23**. Considering that guessing at random leads to a Brier score of **0.25**, it's clear that this feature alone is not very good 😬.

By starting with a simple baseline, it clearly highlighted that more complex patterns were at play. We went ahead to developed another 42 features, in preparation for utilising more complex algorithms, and machine learning models, that might have a better chance.

It was time to move on to the Experiments.

## Models & Machine Learning Experiments

For the models, we opted for simple Neural Networks (NN). To determine which level of complexity would be best, we created three different NNs, with an increasing number of layers and hyper-parameters.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import brier_score_loss
from tensorflow.keras.optimizers import Adam

def create_model_1(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_model_2(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_model_3(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model
```

The next step was running the experiments :alembic:!

### What is an Experiment?

In Fabric, an Experiment allows us to try different hyper-parameters, for each model training run. This set of hyper-parameters, along with the final model score, would be logged, and this information would be available for each run. Once enough runs have been completed, the final model scores can be compared, so that the best version of each model can be selected.

Creating an Experiment in Fabric can be done via the UI or directly from a Notebook. The Experiment is essentially a wrapper for [MLFlow Experiments](https://mlflow.org/). One of great things about using Experiments in Fabric is that the results can be shared with others. The makes it possible to collaborate and allow others to participate in experiments, either writing code to run experiments, or analysing the results.

### Creating an Experiment

Using the UI, to create an Experiment simply select Experiment from the **+ New** button, and choose a name.

![Creating an Experiment using the UI. Shows mouse hovering experiment, with + New dropdown open](./images/exp/exp-1.png "Fig. 6 - Creating an Experiment using the UI")

When training each of the models, the hyper-parameters are logged with the experiment, as well as the final score. Once completed we can see the results in the UI, and compare the different runs to see which model performed best.

![Comparing different runs](./images/exp/exp-4.png "Fig. 7 - Comparing different runs")

After that we can select the best model and use it to make the final prediction. When comparing the three models, the best Brier score was **0.19**, a slight improvement :tada:!

## Conclusion

After loading and analysing data from this year's US major college basketball tournament, and creating a dataset with relevant features, we were able to predict the outcome of the games using a simple Neural Network. The experiments were used to compare the performance of different models. Finally, the best performing model was selected to carry out the final prediction.


In the next post we will go into detail on how we created the features using pyspark. Stay tuned for more! :wave: