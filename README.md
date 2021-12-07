# Using Deep Learning to identify exoplanets from K2 data

Ridhee Gupta, Ryan Moffat, Adam Vert, and Grady Seaward

## Original Paper

The original paper can be read [here](https://iopscience.iop.org/article/10.3847/1538-3881/ab0e12/pdf)

## Project Overview

You can read our proposal presentation [here](https://docs.google.com/presentation/d/1n5lgZY7tB3W2rpewDq_Xp8KH4aIqzklKJyh-J823Fi0/edit#slide=id.p) and our final presentation [here](https://docs.google.com/presentation/d/1uz8E7PzEQqWeeWo50dh50Ruv3hA1q_9qeIvpgF7o_Ac/edit#slide=id.g1066c7fd5ca_17_0). 

![a](assets/NNDiagramUpdated.png)

The goal of this project was to perform a binary classification of Threshold Crossing Events (TCEs) on data from the K2 dataset.
We recreated the network architecture and tested it on the publicly available dataset from the original paper. 

Because this dataset differs from the one used in the paper, we implemented a grid search of three hyperparameters
(learning rate, batch size, and oversampling rate), and analyzed how these hyperparameters affect our overall performance.

To run over a set of hyperparameters, use

python grid_search.py


To generate data from a set of hyperparameters, run

python predictions_maker.py


## File Structure

project

    |-- README.md
    |-- Histogram.py
    |-- auc_curve.py
    |-- create_datasets.py
    |-- networks.py
    |-- predictions_maker.py
    |-- grid_search.py
    |-- K2_candidates.csv
    |-- predictions.npy
        └─── Assets
    |-- outputs
        └─── gridsearch_plots
    |-- NpyData
        └───TestData
        └───TrainData
        └───ValData
    |-- TfRecordsToNpy
