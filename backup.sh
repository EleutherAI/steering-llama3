#!/bin/bash

cd artifacts
tar -cz responses > "../responses_$1.tgz"
cd ..

scp "responses_$1.tgz" adam-ord:src/steering-llama3/
