# !/bin/bash

cd ..
rm build/ -rf
python3 setup.py build
cd tests
