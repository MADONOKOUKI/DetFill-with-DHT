


# # python3 main.py --config configs/dot_proposed.yaml --train --sample_at_start --save_top --gpu_ids 5,6,7,8,9
# # python3 main.py --config configs/dot_proposed.yaml --train --sample_at_start --save_top --gpu_ids 5
# # python3 main.py --config configs/dot_proposed_real.yaml --train --sample_at_start --save_top --gpu_ids 0,1,2,3,4,5,6,7,8,9 > output.txt
# # python3 main.py --config configs/dot_proposed_real.yaml --train --sample_at_start --save_top --gpu_ids 0,1,2,3,4,5,6,7,8,9 > output2.txt
# # python3 main.py --config configs/dot_proposed_real_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_real/checkpoint/top_model_epoch_200.pth --sample_to_eval --save_top --gpu_ids 3 --sample_ratio 0.02
# # 0,1,2,3,4,5,6,7,8,9 > output3.txt
# # python3 main.py --config configs/dot_proposed_real_200epoch.yaml --train --sample_at_start --save_top --gpu_ids 0,1,2,3,4,5,6,7,8,9 > output4.txt
# # python3 main.py --config configs/dot_proposed.yaml --train --sample_at_start --save_top --gpu_ids 0,1,2,3,4
# # python3 main.py --config configs/dot_proposed.yaml --train --sample_at_start --save_top --gpu_ids 1 > output.txt



# python3 main.py --config configs/dot_proposed_real_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_real/checkpoint/top_model_epoch_200.pth --sample_to_eval --save_top --gpu_ids 7 --sample_ratio 0.02 --sketch_type 0
# python3 main.py --config configs/dot_proposed_real_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_real/checkpoint/top_model_epoch_200.pth --sample_to_eval --save_top --gpu_ids 7 --sample_ratio 0.02 --sketch_type 1
# python3 main.py --config configs/dot_proposed_real_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_real/checkpoint/top_model_epoch_200.pth --sample_to_eval --save_top --gpu_ids 7 --sample_ratio 0.02 --sketch_type 2

# python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/top_model_epoch_200.pth --sample_to_eval --save_top --gpu_ids 4 --sample_ratio 0.02 --sketch_type 0
# python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/top_model_epoch_200.pth --sample_to_eval --save_top --gpu_ids 4 --sample_ratio 0.02 --sketch_type 1
# python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/top_model_epoch_200.pth --sample_to_eval --save_top --gpu_ids 4 --sample_ratio 0.02 --sketch_type 2

python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 4 --sample_ratio 0.02 --sketch_type 0
python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 4 --sample_ratio 0.02 --sketch_type 1
python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 4 --sample_ratio 0.02 --sketch_type 2
