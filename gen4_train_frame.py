import os

# 固定パラメータ
gpu_ids = [0]
batch_size_per_gpu = 12
train_workers_per_gpu = 6
eval_workers_per_gpu = 2
mdl_cfg = "tiny"  # MDL_CFGの値を指定
data_dir = "./datasets/pre_gen4"  # DATA_DIRの値を指定

sampling = "random"
input_channels = 3  # 入力チャンネル数
event_frame_dts = [50]  # 必要に応じて値を追加

gpu_ids_str = ",".join(map(str, gpu_ids))


# ループ処理
for dt in event_frame_dts:
    command = f"""
    python3 train.py model=rnndet dataset=gen1 dataset.path={data_dir} wandb.project_name=part2_SAST_gen4_frame_{dt} \
    wandb.group_name=1mpx +experiment/gen4={mdl_cfg}.yaml hardware.gpus="[ {gpu_ids_str} ]" \
    batch_size.train={batch_size_per_gpu} batch_size.eval={batch_size_per_gpu} \
    hardware.num_workers.train={train_workers_per_gpu} hardware.num_workers.eval={eval_workers_per_gpu} \
    dataset.ev_repr_name="'event_frame_dt={dt}'" model.backbone.input_channels={input_channels} validation.val_check_interval=10000\
    dataset.train.sampling={sampling} 
    """
    print(f"Running command for gen4 event_frame_dt={dt}")
    os.system(command)  # 実際にコマンドを実行