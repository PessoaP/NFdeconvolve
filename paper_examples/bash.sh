#!/bin/bash

mkdir -p datasets
mkdir -p graphs
mkdir -p models
mkdir -p tables



python make_data.py

n_values=(10000 3160 1000 316 100)
methods=("SUM_Bayes.py" "SUM_NPBayes.py" "SUM_Normflow.py" "PROD_Bayes.py" "PROD_NPBayes.py" "PROD_Normflow.py")
methods=("SUM_Normflow.py" "PROD_Normflow.py")
methods=("PROD_Normflow.py")
for method in "${methods[@]}"
do
    for i in {3..9}
    do
        for n in "${n_values[@]}"
        do
            echo "Running $method with n = $n, i = $i"
            python "$method" "$n" "$i"
        done
    done
done

python make_figures.py
python figs12.py

python oblation.py
python robustness.py
