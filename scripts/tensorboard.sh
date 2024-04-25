nohup tensorboard --logdir ./output --host $(hostname -i) --port 8040 > tensorboard.log 2>&1 &
# nohup tensorboard --logdir ./output --host=localhost --port 8040 > tensorboard.log 2>&1 &
