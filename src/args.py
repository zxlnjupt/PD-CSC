import argparse


# path = '../dataset'
# folder = 'DFT321_data_len64'
# name = 'data_path_len64.json'
def parse_args():
    parser = argparse.ArgumentParser()
    # 文件路径相关
    parser.add_argument('--root_path', type=str, default="/root/autodl-tmp/chujue/dataset")
    # parser.add_argument('--data_path', type=str, default="Partition_2345_norm_DFT321_data_len64")
    # parser.add_argument('--json_name', type=str, default="Partition_2345_norm_DFT321_data_len64.json")
    # parser.add_argument('--val_data_path', type=str, default="Partition_1_norm_DFT321_data_len64")
    # parser.add_argument('--val_json_name', type=str, default="Partition_1_norm_DFT321_data_len64.json")

    #训练和验证数据集（fold1）
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--json_name', type=str, default="file_list.json")
    parser.add_argument('--val_data_path', type=str, default="")
    parser.add_argument('--val_json_name', type=str, default="data_path_list.json")
    #用一条数据进行测试
    # 训练数据相关参数
    parser.add_argument('--target_sample_hz', type=int, default=2_800)      
    parser.add_argument('--data_length', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    # 训练过程相关参数（重要）
    # 将训练上限改为epoch数量
    parser.add_argument('--num_epochs', type=int, default=1000)  # 总训练epoch数
    # 保留num_train_steps参数以防其他脚本需要，但不再用于控制训练上限
    parser.add_argument('--num_train_steps', type=int, default=2_000_000, help='不再用于控制训练上限，保留以防其他脚本需要')
    # parser.add_argument('--num_train_steps', type=int, default=5_000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--grad_accum_every', type=int, default=4)
    # 训练过程保存
    parser.add_argument('--save_model_every', type=int, default=100)  # 每多少步保存一次模型（保留以防需要）
    parser.add_argument('--save_results_every', type=int, default=100)  # 每多少步保存一次结果（保留以防需要）
    # 基于epoch的保存参数
    parser.add_argument('--save_model_every_epoch', type=int, default=10)  # 每多少个epoch保存一次模型
    parser.add_argument('--log_losses_every', type=int, default=1)
    parser.add_argument('--apply_grad_penalty_every', type=int, default=4)
    # 优化器相关参数
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--discr_max_grad_norm', type=float, default=None)
    # EMA相关参数（别动）
    parser.add_argument('--ema_beta', type=float, default=0.995)
    parser.add_argument('--ema_update_after_step', type=int, default=500)
    parser.add_argument('--ema_update_every', type=int, default=10)

    args = parser.parse_args()
    return args
