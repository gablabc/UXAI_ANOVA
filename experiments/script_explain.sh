# !/bin/bash

# The seed is provided as argument to allow runs in parallel
seed=$1
for data in "kin8nm" "bike" "adult_income" "marketing" "default_credit" "california"
do
    for model in "gbt" "rf"
    do
        # FD-Trees
        for partition in "gadget-pdp" "l2coe" "pfi"
        do  
            python3 3_0_train_fdtree.py --name=$data --model_name=$model --type=$partition --random_state=$seed --save
            python3 3_1_compute_explanations.py --name=$data --model_name=$model --type=$partition --random_state=$seed --save
            python3 3_2_global_importance.py --name=$data --model_name=$model --type=$partition --random_state=$seed --save
            python3 3_3_local_attribution.py --name=$data --model_name=$model --type=$partition --random_state=$seed --save
        done


        # Two baselines
        # CART
        python3 3_0_train_fdtree.py --name=$data --model_name=$model --type=cart --random_state=$seed --relative_decrease=1 --save
        python3 3_1_compute_explanations.py --name=$data --model_name=$model --type=cart --random_state=$seed --relative_decrease=1 --save
        python3 3_2_global_importance.py --name=$data --model_name=$model --type=cart --random_state=$seed --save
        python3 3_3_local_attribution.py --name=$data --model_name=$model --type=cart --random_state=$seed --save
        
        # Fully grown random trees
        python3 3_0_train_fdtree.py --name=$data --model_name=$model --type=random  --negligible_imputity=0 \
                                    --relative_decrease=2 --samples_leaf=0 --random_state=$seed --save
        python3 3_1_compute_explanations.py --name=$data --model_name=$model --type=random --random_state=$seed --save
        python3 3_2_global_importance.py --name=$data --model_name=$model --type=random --random_state=$seed --save
        python3 3_3_local_attribution.py --name=$data --model_name=$model --type=random --random_state=$seed --save
    done
done