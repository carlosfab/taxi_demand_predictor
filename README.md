![Banner](./header.gif)

# Taxi Demand Prediction 🚗

<img src="https://img.shields.io/badge/work%20in%20progress-FF103F" />
<a href="http://linkedin.com/in/carlos-melo-data-science/" alt="linkedin"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white" /></a> 
<a href="http://twitter.com/carlos_melo_py" alt="twitter"> <img src="https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white" /></a> 

Predicting the user demand for a ride-sharing company for the upcoming hour to optimize fleet distribution and maximize revenue. This README will guide you through the various facets of the project, from setup to contribution.

> **Note**: This project is currently a work in progress. I will be making significant updates throughout this week.

# Quickstart/Demo

_To be updated with actual project demonstration or guide._

# Table of Contents

- [Taxi Demand Prediction 🚗](#taxi-demand-prediction-🚗)
- [Quickstart/Demo](#quickstartdemo)
- [Table of Contents](#table-of-contents)
- [Code Structure](#code-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Contribute](#contribute)
- [License](#license)

# Code structure

The project follows an organized directory structure, ensuring clarity, modularity, and ease of navigation. Here is a breakdown of the structure:

```bash
.
├── README.md                     - provides an overview of the project

│   ├── raw                       - contains the raw, unprocessed ride data.
│   │   ├── rides_2022-01.parquet 
│   │   ├── rides_2022-02.parquet 
│   │   └── ...
│   └── transformed               - contains datasets that have undergone some form of processing
│       ├── tabular_data.parquet  
│       ├── ts_data_rides_2022_01.parquet  
│       └── validated_rides_2022_01.parquet 
│       └── ... 
├── models                        - any machine learning models.
├── notebooks                     - exploratory and developmental Jupyter notebooks.
├── pyproject.toml                - project metadata and dependencies
├── scripts                       - scripts for automation, data collection, and other utilities.
├── src                           - directory containing reusable code, functions, and classes.
└── tests                         - test scripts for functionalities
```

# Installation
[(Back to top)](#table-of-contents)

To get started, you'll need to clone this repository and set up the environment:

```shell
git clone <your-repo-link>
cd taxi_demand_predictor
poetry install
poetry shell
```

