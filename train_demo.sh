# start
#python train.py --opt options/train/base7.yaml --name base7_D4C28_bs16ps64_lr1e-3 --scale 3 --ps 64 --bs 16 --lr 1e-3 --gpu_ids 0 
python train.py --opt options/train/base7_qat.yaml --name base7_D4C28_bs16ps64_lr1e-3_qat --scale 3 --ps 64 --bs 16 --lr 1e-3 --gpu_ids 1 --qat --qat_path experiment/base7_D4C28_bs16ps64_lr1e-3/best_status
