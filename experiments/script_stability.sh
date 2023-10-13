# !/bin/bash

for data in "kin8nm" "bike" "adult_income" "marketing" "compas" "california"
do
    for model in "gbt" "rf"
    do 
        python3 3_0_train_fdtree.py --name=$data --model_name=$model --type=l2coe --random_state=0 --background_size=1000
    done
done