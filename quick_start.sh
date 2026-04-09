source .venv/bin/activate
python dextoolbench/eval_interactive.py \
--config-path pretrained_policy/config.yaml \
--checkpoint-path pretrained_policy/model.pth \
--port 8000