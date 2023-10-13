# !/bin/bash

# RF on Adult
for seed in {0..4}
do
    # Random Forests
    python3 1_2_train_trees.py --model_name=rf --name=bike --max_depth=18 --max_samples=0.85 --min_samples_leaf=2 \
                               --n_estimators=350 --random_state=$seed --save
    python3 1_2_train_trees.py --model_name=rf --name=adult_income --criterion=gini --max_depth=18 --max_samples=0.9\
                               --min_samples_leaf=4 --n_estimators=100 --random_state=$seed --save
    python3 1_2_train_trees.py --model_name=rf --name=california --max_depth=25 --max_samples=0.99\
                               --min_samples_leaf=2 --n_estimators=200 --random_state=$seed --save
    python3 1_2_train_trees.py --model_name=rf --name=compas --max_depth=4 --max_features=0.5 --max_samples=0.8\
                               --min_samples_leaf=6 --n_estimators=140 --random_state=$seed --save
    python3 1_2_train_trees.py --model_name=rf --name=marketing --criterion=gini --max_depth=26 --max_samples=0.7\
                               --min_samples_leaf=1 --n_estimators=200 --random_state=$seed --save
    python3 1_2_train_trees.py --model_name=rf --name=kin8nm  --max_depth=17 --max_features=0.7 --max_samples=0.99\
                               --min_samples_leaf=1 --n_estimators=400 --random_state=$seed --save
    python3 1_2_train_trees.py --model_name=rf --name=default_credit --criterion=entropy  --max_depth=18 \
                               --max_features=0.75 --max_samples=0.5 --min_samples_leaf=3 --n_estimators=250 \
                               --random_state=$seed --save
    
    # Gradient Boosted Trees
    python3 1_2_train_trees.py --model_name=gbt --name=bike --subsample=0.6 --n_estimators=350 --min_samples_leaf=14\
                               --max_depth=9 --learning_rate=0.04 --random_state=$seed --save
    python3 1_2_train_trees.py --model_name=gbt --name=adult_income --subsample=0.8 --n_estimators=450 --min_samples_leaf=29\
                               --max_depth=3 --learning_rate=0.1421 --random_state=$seed --save
    python3 1_2_train_trees.py --name=california --model_name=gbt --learning_rate=0.06 --max_depth=7 --min_samples_leaf=9 \
                               --n_estimators=450 --subsample=0.8 --random_state=$seed --save
    python3 1_2_train_trees.py --name=compas --model_name=gbt --learning_rate=0.1667 --max_depth=2 --min_samples_leaf=8 \
                               --max_features=0.25 --n_estimators=130 --subsample=0.6 --random_state=$seed --save
    python3 1_2_train_trees.py --name=marketing --model_name=gbt --learning_rate=0.04316 --max_depth=9 --min_samples_leaf=15 \
                               --max_features=0.25 --n_estimators=200 --subsample=0.9 --random_state=$seed --save
    python3 1_2_train_trees.py --name=kin8nm --model_name=gbt --learning_rate=0.03667 --max_depth=9 --min_samples_leaf=5 \
                               --max_features=0.75 --n_estimators=400 --subsample=0.6 --random_state=$seed --save
    python3 1_2_train_trees.py --name=default_credit --model_name=gbt --learning_rate=0.132 --max_depth=3 --min_samples_leaf=12 \
                               --max_features=0.75 --n_estimators=250 --subsample=0.8 --random_state=$seed --save

done

