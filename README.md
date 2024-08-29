# CML Gurobi

## Objective

Cloudera Machine Learning (CML) is a platform designed to help organizations build, deploy, and manage machine learning models at scale. It is part of Cloudera’s suite of enterprise data platforms and solutions, focusing on providing a robust environment for data scientists, analysts, and engineers to collaborate on end-to-end machine learning workflows.

PyGurobi is a Python interface for the Gurobi Optimizer, a powerful and widely used solver for mathematical optimization problems. Gurobi is known for its high performance in solving a variety of optimization problems, including linear programming (LP), quadratic programming (QP), mixed-integer programming (MIP), and others.

In this tutorial you will use PyGurobi on CML in order to optimize product prices and maximize enterprise revenue.

## Requirements

The following are required in order to reproduce this example:

* CML Workspace in AWS, Azure, OCP or ECS.
* Basic knowledge of Python for Machine Learning including Sci-Kit Learn, Spark, Iceberg, and XGBoost.
* Basic familiarity with linear and non linear programming. If you are new to mathematical optimization please visit this link for a [quick introduction](https://web.stanford.edu/group/sisl/k12/optimization/MO-unit3-pdfs/3.4buildingsimplex.pdf).

## Step by Step Instructions

Launch a CML Session with the following runtime and resource profile:

```
Editor: JupyterLab
Kernel: Python 3.10
Edition: Standard
Version: 2024.05
Enable Spark: Spark 3.2 or above
Resource Profile: 2 CPU / 4 GB Mem / 0 GPU
Runtime Image: docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-jupyterlab-python3.10-standard:2024.05.1-b8
```

Open the terminal and install the requirements:

```
pip3 install -r requirements.txt
```

#### Part 0: Data Generation

Run notebook ```00_datagen_iceberg_pyspark.ipynb``` and observe the following:

* A Spark dataframe with 10000 synthetic product price transactions is created.
* The P1 and P2 columns represent the price for two products sold. The N1 column represents the quantity of Product 1 sold.
* The dataframe is stored as an Iceberg table.

#### Part 1: Pricing Optimization with Gurobi

Run notebook ```01_price_optimization_with_competing_products.ipynb``` and observe the following:

* An MLFlow Experiment Context is created with name "Price Optimization Experiment".
* An initial regressor is built to predict prices using the data stored in the Iceberg table. The data is read using PandasOnSpark which is included in the Spark Runtime AddOn by default.
* A Price Optimization model is instantiated with an Objective Function and associated Constraints.
* The model is trained on the data. Its outputs include an optimal price recommendation for the two products, with an associated product quantity for the two products, and finally a revenue estimate. In other words, revenue is maximized at 70347.77 when prices are 400 and 300 for the two products respectively.

#### Part 2: Deploy Optimization Model in an API Endpoint

Run notebook ```02_price_optimization_model_deployment.ipynb``` and observe the following:

* CML APIv2 allows you to programmatically execute actions within CML Workspaces. In this example the API is used to create a small Python Interface to manage model deployments.
* In particular, the interface was used to create a separate CML Project to host an API Endpoint. The API Endpoint is used to allocate a dedicated container for the model and provide an entry point for prediction requests.

Navigate back to the CML workspace and notice a new project named ```CML Project for Optimization Model``` has been created. Open it and notice a new Endpoint has been created in the Model Deployments section.

Open the model deployment and, once it has completed, enter the following sample payload in the Test Request window. Observe the output response.

```
{"p[1]": [354,353,352,351,354,353,312,311,314,313,352,351], "p[2]": [110,120,320,220,101,100,101,260,355,140,300,299], "n[1]": [54,53,112,151,154,153,52,51,4,53,92,71]}
```

## Summary

In this tutorial you used PyGurobi in Cloudera Machine Learning to maximize product revenue by identifying optimal prices and sales quantities for two products.

The PyGurobi library allows you to solve complex linear and nonlinear programming such as the above. Cloudera on Cloud provides the tooling necessary to use libraries such as PyGurobi in an enteprise setting. With CML you can easily leverage Spark on Kubernetes, Runtime Add-Ons, Iceberg, Python, MLFlow, and more, to install and containerize workdloads and machine learning models at scale, without any custom installations.

## Related Articles and Resources

Here are some useful articles about Cloudera Machine Learning (CML) that can help you better understand its features and capabilities:

1. **Cloudera Machine Learning - What You Should Know**: This article on the Cloudera Community provides an overview of CML, explaining how it enables teams to deploy machine learning workspaces that auto-scale to fit their needs using Kubernetes. It highlights key features like cost-saving through auto-suspend capabilities and offers a consistent experience across an organization. The article is a good starting point for understanding CML's role within the Cloudera Data Platform (CDP). [Introduction to CML](https://community.cloudera.com/t5/Community-Articles/Cloudera-Machine-Learning-What-You-Should-Know/ta-p/292935).

2. **How to Use Experiments in Cloudera Machine Learning**: This guide walks through using experiments in CML, which allows users to run scripts with different inputs and compare metrics, particularly useful for tasks like hyperparameter optimization. The article includes practical examples that illustrate how experiments can be applied in real-world. [MLFlow Experiments in CML](https://community.cloudera.com/t5/Community-Articles/How-to-use-Experiments-in-Cloudera-Machine-Learning/ta-p/294554).

3. **Cloudera Machine Learning Documentation**: This hands-on guide from Datafloq provides a detailed checklist for managing CML projects effectively, with a focus on optimizing productivity and data quality. It discusses essential components such as data cleansing and how these contribute to improved decision-making, which is crucial for successful machine learning outcomes. [CML Documentation](https://docs.cloudera.com/machine-learning/cloud/product/topics/ml-product-overview.html).

4. **Getting Started with the Gurobi Python API**: This tutorial provides a comprehensive introduction to using PyGurobi, from creating a model, adding variables and constraints, to setting objectives and optimizing. It explains the use of key functions such as `Model.addVar`, `Model.addConstr`, and `Model.setObjective`, making it an excellent starting point for beginners interested in mathematical optimization using Gurobi in Python [Gurobi Help Center](https://support.gurobi.com).

5. **Python API Overview**: This overview explains the various types of models that can be handled by Gurobi, such as Mixed Integer Linear Programs (MILP), Mixed Integer Quadratic Programs (MIQP), and Non-Linear Programs (NLP). It also covers the environments used within the Gurobi Python interface and provides guidance on solving models, managing multiple solutions, and handling infeasible models. [Pythin API Overview](https://www.gurobi.com/documentation/current/refman/py_python_api_overview.html)

6. **Gurobi Optimizer Python Environment**: This resource outlines how to set up and start using the Gurobi Python environment, including installing the Gurobi package via Anaconda or pip. It also highlights the different types of licenses available, including free academic licenses and evaluation licenses for commercial users. [Gurobi Optimizer Python Environment](https://www.gurobi.com/features/gurobi-optimizer-delivers-rich-python-modeling-environment/)
