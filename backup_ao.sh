#!/bin/bash

echo "Merging SO responses..."

python smartcopy.py "responses_SO_$1.tgz" artifacts/responses

echo "Tarring all responses..."

cd artifacts
tar -cz responses > "../responses_all_$1.tgz"
cd ..

echo "Done!"