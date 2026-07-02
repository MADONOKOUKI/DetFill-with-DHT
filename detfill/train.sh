




python3 main.py --config configs/scribble_proposed_illust_200epoch.yaml --train --sample_at_start --save_top --gpu_ids 0,1,2,3,4,5,6,7,8,9 --port 12356 > output4.txt
python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --train --sample_at_start --save_top --gpu_ids 0,1,2,3,4,5,6,7,8,9 --port 12356 > output4.txt
python3 main.py --config configs/scribble_proposed_real_200epoch_rev.yaml --train --sample_at_start --save_top --gpu_ids 0,1,2,3,4,5,6,7,8,9 --port 12356 > scribble_real.txt
python3 main.py --config configs/dot_proposed_real_200epoch_rev.yaml --train --sample_at_start --save_top --gpu_ids 0,1,2,3,4,5,6,7,8,9 --port 12356 > scribble_real.txt