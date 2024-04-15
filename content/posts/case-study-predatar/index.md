---
title: "Deploying an Anomaly Detection System End-to-End"
summary: "In this case study, we'll be detailing our work with a client to build and deploy an anomaly detection system."
date: 2024-04-15T17:00:43Z
draft: true
showAuthor: true
authors:
  - "martimchaves"
  - "rogernoble"
tags:
  - "Anomaly Detection"
  - "Cyber Security"
series: ["Case Studies"]
series_order: 1
---

## Introduction

In this post, we'll be taking a deep dive into the work that we've made for one our clients - an end-to-end Anomaly Detection System. We'll be going over the client's goal, the state of the system, the solution that we deployed, the overall results and what we learned. 

### The client's goal

The client, X, is a cyber-security company. Among other features, it provides an anomaly detection, whose aim is to detect anomalies in the day to day activities of machines. The assumption under this system is that unusual patterns of activity can be indicative of a security threat. X's goal is not only to ensure that this system has a good performance, but that it also is reliable and resilient against evolving security challenges.

---

## The Anomaly Detection System

This anomaly detection system requires careful engineering of several moving parts and the use of machine learning (ML) models to detect deviations of normal behaviour. 

### Challenges

- Data Volume and Complexity
- Scalability and Resource Constraints
- Integration with Existing Architecture

### Objectives and Requirements

- Good performance (accuracy)
- Reliability
- Future-proofing

---

## Solutions Deployed

To meet the objectives, several steps were carried out to develop the anomaly detection system.

### Assessing the initial performance

Beforing carrying out any improvements, it was crucial to first assess the state of the system.

#### Careful metric selection

It was found that often data would have several spikes - this meant that it made more sense to use the mean squared error, instead of the mean absolute error, as the main metric.

#### Manual Sanity checks

We manually analysed the data, and the predictions of the model that was being used. We found that, due to how the data was being prepared, although the score was quite good, the performance of the model was actually not the best!

#### How can Machine Learning Systems fail silently?

A tricky thing about ML systems is that they can fail silently - in other, they may look like they're working as intended but they aren't. In this case, there were no errors, and the performance metrics showed that the model was, supposedly, making great predictions. But in reality that was not the case. (expand) That's why carefully selecting performance metrics and often doing manual sanity checks is crucial to a healthy ML system.

### Adding a new feature for Anomaly Detection

Besides Bytes Transferred, we added the Deduplication Ratio as another feature to detect anomalies on - in the process, we made sure that adding new features would be possible with minimal effort.

### Experiments

- There were several stages of the Anomaly Detection System that could be tested and experimented with

#### Data preparation

How the data was prepared was one of the main experimentation points. Including zeros for days where there were no measurements, decomposing the time series into its three parts, using all of the features at the same time or separately, adding a day of the week feature, and combinations of these were tested.

#### Models tested

We wanted to assess how complex the model would have to be, and thus we experiment with a classic ARIMA model, a small LSTM, and a larger, more complex, auto-encoder LSTM.

#### Possible Configurations

Considering the previous specifications, there were around 16 possible different configurations, and for each configuration hyperparameter optimisation was carried out.

#### Using MLFlow

MLFlow was the package that was used to track and compare all of these experiments.

#### Drawing Conclusions based on Requirements

When selecting the winner configuration, raw performance was not the only aspect that counted. We wanted the system to be as light as possible, to reduce computation time. We also wanted to reduce the number of days required to use the anomaly detection system.

### Updating the Anomaly Detection System

After selecting the winner configuration, the system was updated - version 2.

### Adding a Logging System

To make the system more robust, we wanted to create a logging system that would allows us to restart the system at any point.

#### The Training Cycle

The training cycle consisted of reading the data and training a model for each VM.

#### The Inference Cycle

The inference cycle consisted of checking which VMs have available models, and carrying out inference for those.

#### Combining the two cycles for a smooth deployment

Altough the training and inference cycles are pretty much independent, they are related, and it was key to get this part right.

---

## Results and Impact
### Performance Improvements
How was performance improved?
### Reliability and Robustness
What is the current reliability of the system?

---

## Analysis and Learnings
### Evaluating the current solution
Making the experiments resumeable + splitting experiments into containers for faster processing
### Learnings and Recommendations
Queue system/Airflow

---

## Conclusion
### Summary
### Future Work
### Testimonials
