---
title: "Anomaly Detection for Backup Security"
summary: "Per-server machine learning models that detect ransomware and data exfiltration from backup metadata."
industry: "Data Protection / Cyber Security"
date: 2026-04-06
draft: false
---

## Data Protection / Cyber Security

A data protection company manages backup infrastructure for thousands of servers across their customer base. They needed a way to automatically detect potentially malicious activity — such as ransomware encryption or unauthorised data exfiltration — by analysing backup metadata. The challenge was that every server has different backup patterns, so a one-size-fits-all approach to anomaly detection wouldn't work.

We designed and built a system that trains an individual machine learning model per server, learning what "normal" looks like for that specific machine's backup behaviour. The system monitors two key signals: the volume of data being transferred and the deduplication ratio. A sudden spike in bytes transferred could indicate data being exfiltrated, while a drop in deduplication ratio can be a sign of ransomware encryption — encrypted data doesn't deduplicate well.

The approach uses LSTM autoencoder neural networks to learn each server's patterns and predict expected values. When actual values deviate significantly from predictions, the system flags an anomaly and generates a threat score on a 1–10 scale based on the severity of the deviation.

A key part of the project was continuous iteration on model quality. Over multiple phases we systematically evaluated and improved every aspect of the pipeline — from data preprocessing and sampling strategies to model architecture and threshold calculations. We introduced rigorous evaluation metrics including Dynamic Time Warping and RMSE alongside standard error measures, which gave us a much clearer picture of whether models were genuinely learning server behaviour rather than just minimising error on paper. This experimental, evidence-driven approach meant the system got measurably better with each iteration.

Beyond the core data science work, we built out the production infrastructure: Apache Airflow for orchestrating training and inference pipelines, S3-compatible object storage for model persistence, Kubernetes for scalable deployment, and automated CI/CD. We also introduced proper MLOps practices including model versioning, quality monitoring, threshold tuning based on statistical measures rather than fixed values, and a structured experimentation framework for evaluating alternative model architectures.

The system runs in production, continuously monitoring backup activity and surfacing threat scores to end users, giving security teams early warning of potential attacks that would otherwise go undetected until it was too late.
