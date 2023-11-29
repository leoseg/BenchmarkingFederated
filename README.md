# MasterthesisBenchmarkFL
Benchmarking of Flower and Tensorflow Federated for gene expression data analysis.

## Datasets used
### First Usecase: Blood Cell Classification
The first use case is a binary classification task where the model should learn to predict if the patient has acute myeloid leukemia (AML).
This use case was investigated with a swarm learning algorithm ([Warnat-Herresthal S., Schultze H. et al., 2020](https://doi.org/10.1038/s41586-021-03583-3)) and analyzed with a regression model ([Warnat-Herresthal .S, Perrakis K. et al., 2020](#AMLlasso)). In the two studies, it has been shown that the data contains valuable information and is suited for this classification task.
The data set used is a gene expression matrix from blood transcriptomes gathered from peripheral blood mononuclear cells (PBMC).
Each sample contains the expression per gene and a label which classifies the patient's condition, in healthy or has AML.
To produce the data set used for benchmarking multiple steps were done from authors of the other studies.
The original data set was formed out of 105 studies in total resulting in 12,029 samples. from the gene expression omnibus (GEO) database and were first divided into three data sets by the technique the data was obtained.
The techniques used were HG-U133A and HG-U133A 2.0 microarray techniques and RNA-seq technique.
The reads obtained by RNA-seq were mapped with the Kallisto aligner against the human reference genome gene code version 27 (GRCh38.p10) ([Warnat-Herresthal S., Perrakis K.. et al., 2020],(https://doi.org/10.1016/j.isci.2019.100780)).
After that all three data sets were normalized independently, the microarray data sets with robust multichip average (RMA) and the RNA-seq data set with the R package DESeq2 using standard parameters.
All genes which were not present in all three data sets were filtered resulting in 12,708 genes.

### Second Usecase: Cell Type Classification
The second use case is a multiclass classification task where the correct cell subtype should be classified from an RNA-seq data set of the human middle temporal gyrus (MTG).
The data set of this use case was analyzed with different machine learning algorithms ([Le H., Peng B., et al., 2022](https://doi.org/10.1371/journal.pone.0275070)) and it has been shown that it is a suitable classification problem for logistic regression.
The data set in the form of an R - Dataframe can be downloaded from Allen Brain Map (https://portal.brain-map.org/atlases-and-data/rnaseq).
This data set was produced by isolating sample nuclei from human (MTG) specimens using Dounce homogenization and fluorescence-activated nuclei sorting.
From those samples, the gene expression matrix was produced with RNA-seq. The resulting data set had 15,928 samples from 8 donors between 24-66 and 50,281 genes.
Cell types were defined by PCA and clustering resulting in 75 distinct cell types. This data set was then downloaded, and the preprocessing scripts out of the repository of the study of Le H., Peng B., et al. 2022 were used, normalizing the data by counts per million (CPM) and log2 transforming it.
Also, all samples which are not assigned to a class by the clustering and all genes with zero expression over all samples were removed, resulting in 15,603 samples and 48,840 genes.
Next, the median gene expression within each cell type (MGECT) was calculated for every gene, and all genes with zero variance over all cell types were excluded, resulting in 13,945 genes.
To further filter the number of genes, the coefficient of variation (CV) was used. This coefficient explains how much the gene expression of a gene differs between different samples.
The lower the value, the less the gene indicates a difference between samples. All genes with a value below 3.5 were excluded, resulting finally in 1,426 genes.
To reduce the number of samples and to make the data set more suitable for benchmarking the influence of class imbalance, only the samples labeled with the five most common use cases were chosen, resulting in 6,931 samples.

## Benchmark setup and configurations

### Benchmark Procedure
Both data sets were preprocessed as described above before using them for the benchmarks.
First the hyperparameters are fine-tuned on the entire data set until we have a so-called "central model" with a robust model performance to be used as baseline.
To benchmark the model performance of this central model K-fold-cross-validation is used. A sequential deep learning model (DL) and a logistic regression model (Log) were benchmarked on both data sets. So in the following for each data set model combination, they will be abbreviated as follows:
- First data set: BloodDL, BloodLog
- Second data set: BrainCellDL, BrainCellLog
This baseline model with the fixxed hyperparameters is then used to benchmark the model performance of the federated learning frameworks for different benchmark scenarios.
For each scenario different numbers of rounds are benchmarked to assess the impact of frequency of weight updates among clients during training. The total number of epochs where held consistent so for example for one round each client would
train for 100 epochs and for two rounds each client would train for 50 epochs. This round configurations ar the same for all benchmarks, and are as follows:
- BloodDL, BrainCellDL, BrainCellLog: 1, 2, 5 ,10
- BloodLog: 1, 2, 4, 8
#### Number of clients
In the first benchmark scenario the data set is distributed among different numbers of clients to simulate real-world scenarios and investigate the influence of data heterogeneity on model performance.
The dataset is divided into n partitions, where n is the number of clients. Each partition has the same size and class distribution as the whole dataset.
The data on each client is then split into training and test data with a ratio of 80:20. For evaluation of the model performance the test data of all clients is combined and
the aggregated FL model is evaluated on this test data. For this scenario there were also computational resources metrics recorded. The benchmarks where done with 3, 5, 10 and 50 clients.
#### Different Class distribution
In the second benchmark scenario each client has different class distributions. The number of clients is set to the number of classes and then starting with an equal class distribution
at each client, each the class distribution is changed by increasing the number of samples of one class and decreasing the number of samples of another class. This is done for each client with another class.
For the most extreme configuration each client has only samples of one class. The evaluation of the model performance is here done with a seperated test data set with equal class distribution.
For the Blood data set the benchmarks were done with 50\%, 60\%, 70\%, 80\%, 90\% and 100\% of the samples of one class and for the Brain Cell data set with 50\%, 60\%, 70\%, 80\%, 90\%, 95\% and 100\% of the samples of the chosen class.
For the Brain Cell data set the benchmarks were done with 20\%, 40\%, 60\%, 70\%, 80\%, 90\% and 100\% of the samples of the chosen class.

#### Differential Privacy
In the third benchmark scenario the influence of differential privacy on model performance is investigated. Before aggregating the clients model parameters, gaussian noise is
added to the client weights. The number of clients is fixxed to 5 and the model performance evaluation is done like in the number of clients benchmark.
The noise parameters 0.01, 0.03, 0.05, 0.07, 0.085, 0.1 were used.
#### Implementations details

Models are implemented using Keras, with default settings and federated algorithms from both federated learning frameworks.
Specifics of implementation in TensorFlow Federated (TFF) and Flower frameworks, including the federated averaging strategy and customization of source code to record system metrics.
Metrics are recorded using WandB, with separate groups for each configuration and model. The network traffic was measured with tshark.
To ensure resource parity among different frameworks and the central model,
the memory is set to 50 GB for all benchmarks, and each training process is bound to a different CPU only used by itself.
All benchmarks are done on the scientific computing cluster of the university of Leipzig.
The computational resources metric benchmarks were done on the clara cluster.
Each node there has multiple CPUs with an AMD(R) EPYC(R) 7551P@ 2.0GHz - Turbo 3.0GHz processor and 31 Gigabyte RAM.
For the benchmark each client and server was restricted to use only one CPU. The network is 100 Gbit/s Infiniband.
No GPU is used during the benchmarks.

### Model parameters
- Batch size always : 512
- Loss: CrossEntropy
- Deep learning architecture
  - Dense(256, activation='relu')
  - Dropout(0.4)
  - Dense(512, activation='relu')
  - Dropout(0.15)
  - Dense(256, activation='relu')
  - Dropout(0.15)
  - Dense(256, activation='relu')
  - Dropout(0.15)
  - Dense(128, activation='relu')
  - Dropout(0.15)
  - Dense(128, activation='relu')
  - Dropout(0.15)
  - Dense(64, activation='relu')
  - Dropout(0.15)
  - Dense(64, activation='relu')
  - Dropout(0.15)
  - Dense(32, activation='relu')
  - Dropout(0.15)
  - Outputlayer
- Blood dataset:
  - Deep Learning:
    - Optimizer: Adam
    - L2 regularization: 0.005
    - Number Epochs: 70
  - Logistic Regression:
    - Optimizer: SGD
    - L2 regularization: 0.001
    - Epochs: 8
- Brain Cell dataset:
  - Deep Learning:
    - Optimizer: Adam
    - L2 regularization: 0.005
    - Number Epochs: 30
  - Logistic Regression:
    - Optimizer: SGD
    - L2 regularization: 1.0
    - Epochs: 10

### Metrics Used

- Accuracy: Percentage of correctly predicted values matching the label.
- AUC (Area Under Curve): Measures the classifier's ability to distinguish between negative and positive examples.
- Memory Consumption: Measured in Gigabytes for both the client and server during model training, is calculated by summing up the memory consumption each second over the whole duration.
- Training Time: Time taken for training the model in seconds at one client (per round) or in case of the central learning for the central model.
- Round Time: Time taken for a training round  in seconds.
- Data Transmission: Amount of data sent from client to server and received from server at client, measured in Megabytes




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
