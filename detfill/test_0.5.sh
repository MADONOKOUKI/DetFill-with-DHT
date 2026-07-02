


# python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 6 --sample_ratio 0.5 --sketch_type 0
# python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 6 --sample_ratio 0.5 --sketch_type 1
# python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 6 --sample_ratio 0.5 --sketch_type 2

# python3 main.py --config configs/scribble_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 0
# python3 main.py --config configs/scribble_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 1
# python3 main.py --config configs/scribble_proposed_illust_200epoch.yaml --resume_model results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 2

# python3 main.py --config configs/dot_proposed_real_200epoch_rev_nop.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb_nop/real/dot --resume_model results/dataset_name_nop/BBDM_dot_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 7 --sample_ratio 0.5 --sketch_type 0
# python3 main.py --config configs/dot_proposed_real_200epoch_rev_nop.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb_nop/real/dot --resume_model results/dataset_name_nop/BBDM_dot_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 7 --sample_ratio 0.5 --sketch_type 1
# python3 main.py --config configs/dot_proposed_real_200epoch_rev_nop.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb_nop/real/dot --resume_model results/dataset_name_nop/BBDM_dot_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 7 --sample_ratio 0.5 --sketch_type 2


# python3 main.py --config configs/scribble_proposed_real_200epoch_rev.yaml --resume_model results/dataset_name/BBDM_scribble_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 0
# python3 main.py --config configs/scribble_proposed_real_200epoch_rev.yaml --resume_model results/dataset_name/BBDM_scribble_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 1
# python3 main.py --config configs/scribble_proposed_real_200epoch_rev.yaml --resume_model results/dataset_name/BBDM_scribble_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 2
# python3 main.py --config configs/dot_proposed_real_200epoch_rev.yaml --resume_model results/dataset_name/BBDM_dot_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 0
# python3 main.py --config configs/dot_proposed_real_200epoch_rev.yaml --resume_model results/dataset_name/BBDM_dot_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 1
# python3 main.py --config configs/dot_proposed_real_200epoch_rev.yaml --resume_model results/dataset_name/BBDM_dot_real/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 2


python3 main.py --config configs/dot_proposed_illust_200epoch_nop.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb_nop/illust/dot/det --resume_model results/dataset_name_nop/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 0
python3 main.py --config configs/dot_proposed_illust_200epoch_nop.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb_nop/illust/dot/det --resume_model results/dataset_name_nop/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 1
python3 main.py --config configs/dot_proposed_illust_200epoch_nop.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb_nop/illust/dot/det --resume_model results/dataset_name_nop/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 2
python3 main.py --config configs/dot_proposed_illust_200epoch_nop.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb_nop/illust/dot/det --resume_model results/dataset_name_nop/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.00 --sketch_type 0
python3 main.py --config configs/dot_proposed_illust_200epoch_nop.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb_nop/illust/dot/det --resume_model results/dataset_name_nop/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.00 --sketch_type 1
python3 main.py --config configs/dot_proposed_illust_200epoch_nop.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb_nop/illust/dot/det --resume_model results/dataset_name_nop/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.00 --sketch_type 2

python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb/illust/dot/det --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 0
python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb/illust/dot/det --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 1
python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb/illust/dot/det --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.5 --sketch_type 2
python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb/illust/dot/det --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.00 --sketch_type 0
python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb/illust/dot/det --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.00 --sketch_type 1
python3 main.py --config configs/dot_proposed_illust_200epoch.yaml --result_path /scratch/madono/main_exp_main_exp_felzenszwalb/illust/dot/det --resume_model results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth --sample_to_eval --save_top --gpu_ids 2 --sample_ratio 0.00 --sketch_type 2


