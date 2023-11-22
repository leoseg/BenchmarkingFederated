# MasterthesisBenchmarkFL
Benchmarking of Flower and Tensorflow Federated for gene expression data analysis.

## Datasets used

## Benchmark setup and configurations

## Data format
In the BenchmarkData folder in the data folder are the results saved in Json format extracted
and summarized from wandb. There are two different formats, the one for the data of the central model
and the one for the data for the federated model. The data for the central model is saved in the following format:
```
{
    "metric name": Array with the metric values for each repeat
}
```
The data for the federated model is saved in the following format:
```
{
    [Array with one element for each number of rounds configuration] {
        "framework_group": {
            "metric name": Array with the metric values for each repeat
        }
    }
}
```
The data for the system metrics is saved in the json with the names "scenario_metrics_{usecase}_system_network".
The data for the model metrics is saved in the json with the names "scenario_metrics_{usecase}_balanced" or
"scenario_metrics_{usecase}_unweighted". The balanced metrics are benchmarking the model performace for different number
of clients and the unweighted metrics are benchmarking the model performance for different class distribution.
