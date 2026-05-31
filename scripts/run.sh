#!/bin/bash
set -e

cd /home/user

bash scripts/evaluate_train.sh /home/user/outputs/cursor_ranking.json /home/user/outputs/cursor_ranking_eegnet_n500.json
bash scripts/evaluate_test.sh /home/user/outputs/cursor_ranking.json /home/user/outputs/cursor_ranking_eegnet_n500.json
