#!/bin/bash

printf "Training detectors...\n"
python3 train_detector.py

printf "\n\nAttackGAN: DT\n"
python3 main.py 2

printf "\n\nAttackGAN: RF\n"
python3 main.py 0

printf "\n\nAttackGAN: SVM\n"
python3 main.py 1

printf "\n\nAttackGAN: NB\n"
python main.py 3

printf "\n\nAttackGAN: DNN\n"
python3 main.py 4
