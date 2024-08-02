#!/bin/bash

echo "Tarring SO responses..."

cd artifacts
tar -cz responses > "../responses_SO_$1.tgz"
cd ..

echo "Copying to adam-ord..."

scp "responses_SO_$1.tgz" adam-ord:src/steering-llama3/
echo "Done!"