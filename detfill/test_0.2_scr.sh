
python3 main.py --config configs/scribble_proposed_real_200epoch_rev.yaml --resume_model results/dataset_name/BBDM_scribble_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 3 --sample_ratio 0.2 --sketch_type 0
python3 main.py --config configs/scribble_proposed_real_200epoch_rev.yaml --resume_model results/dataset_name/BBDM_scribble_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 3 --sample_ratio 0.2 --sketch_type 1
python3 main.py --config configs/scribble_proposed_real_200epoch_rev.yaml --resume_model results/dataset_name/BBDM_scribble_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 3 --sample_ratio 0.2 --sketch_type 2


python3 main.py --config configs/scribble_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 3 --sample_ratio 0.2 --sketch_type 0
python3 main.py --config configs/scribble_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 3 --sample_ratio 0.2 --sketch_type 1
python3 main.py --config configs/scribble_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 3 --sample_ratio 0.2 --sketch_type 2

