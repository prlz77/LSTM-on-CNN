#!/bin/bash
th ./tests/test_loader.lua
if [[ $? == 1 ]]; then
    echo "Error in data provider."
    exit 1
fi
th LSTM.lua --trainPath tests/toydata2.h5 --valPath ./tests/toydata2.h5 --rho 5 --hiddenSize 8 --depth 3 --dropoutProb 0 --plotRegression 1 --maxEpoch 2
if [[ $? == 1 ]]; then
    echo "Error in LSTM script."
    exit 1
fi
echo "Tests passed."