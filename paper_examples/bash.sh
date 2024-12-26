#!/bin/bash

python make_data.py

#echo "filename,method,N,KL_MAP,KL_reconstruct" > "report.csv"
#echo "filename,method,N,KL_MAP,KL_reconstruct" > "report_prod.csv"


n_values=(10000 3160 1000 316 100)
methods=("SUM_Bayes.py" "SUM_NPBayes.py" "SUM_Normflow.py" "PROD_Bayes.py" "PROD_NPBayes.py" "PROD_Normflow.py")
methods=("PROD_NPBayes.py" "SUM_NPBayes.py")



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