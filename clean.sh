#!/bin/bash
rm -rf ./data/process ./data/featurized/ ./data/processed/ ./models ./.dvc /tmp/dvc_demo ./data/test/ ./metrics
rm ./data/.gitignore ./data/initial_data.csv.dvc dvc.yaml dvc.lock
# git rm -r --cached 'data/initial_data.csv'