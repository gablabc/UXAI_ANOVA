# !/bin/bash

cd models
for data in `ls`
do 
    if [ $data = "sweeps" ]
    then 
        echo "Skip"
    else
        cd $data
        for folder in `ls`
        do
            cd $folder
            pwd
            rm *.npy
            rm *.txt
            rm l2coe*
            rm random*
            rm gadget-pdp*
            rm pfi*
            ls
            cd ..
        done
        cd ..
    fi
done
