# Titanic ML Pipeline with Apache Airflow and MLflow

## Overview

This project implements an **end-to-end Machine Learning pipeline** for predicting passenger survival on the Titanic dataset. The pipeline is orchestrated using **Apache Airflow** and uses **MLflow** for experiment tracking and model registry.

The system demonstrates key **MLOps concepts**, including automated data processing, experiment tracking, model evaluation, conditional workflow branching, and model registration.

The dataset used is the Titanic dataset from Kaggle:
https://www.kaggle.com/datasets/yasserh/titanic-dataset

---

# System Architecture

The pipeline integrates **Apache Airflow** and **MLflow**.

### Apache Airflow
Airflow orchestrates the machine learning workflow using a **Directed Acyclic Graph (DAG)**. Each stage of the pipeline is represented as a task in the DAG.

Airflow manages:
- Workflow scheduling
- Task dependencies
- Parallel task execution
- Failure handling and retries
- Conditional branching

### MLflow
MLflow is used for **experiment tracking and model management**.

MLflow logs:
- Model parameters
- Evaluation metrics
- Model artifacts
- Experiment runs

MLflow Model Registry stores approved models for deployment.

---

# Pipeline Workflow

The DAG executes the following tasks:
