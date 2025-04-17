!/usr/bin/env bash

python main.py --plot --plot_results roc --dataset_name location
python main.py --plot --plot_results roc --dataset_name fmnist
python main.py --plot --plot_results roc --dataset_name stl10
python main.py --plot --plot_results roc --dataset_name cifar10
python main.py --plot --plot_results roc --dataset_name cifar100
python main.py --plot --plot_results roc --dataset_name texas
python main.py --plot --plot_results roc --dataset_name adult
python main.py --plot --plot_results roc --dataset_name purchase

