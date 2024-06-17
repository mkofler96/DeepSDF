nohup python3 -u train_deep_sdf.py --experiment experiments/simple_geom --batch_split 5 > logScriptNohup.log &
tail -f logScriptNohup.log