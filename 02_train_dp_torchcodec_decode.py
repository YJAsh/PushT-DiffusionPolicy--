from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
from torch.utils.data import DataLoader

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

#一、准备数据集 与 数据集相关的参数设置

#pusht数据集的演示视频帧率为 10 fps，演示动作序列长度为 64 帧。
#我们希望模型学习从两帧观察中预测接下来的 64 帧动作。
fps=10
horizon=16
first_offset=-1 #从当前帧的上一帧开始

# 计算每个样本的 观察帧 和 动作帧 的时间戳
obs_times=[first_offset/fps,0.0]
act_times=[i/fps for i in range(first_offset, first_offset+horizon)]

#数据集参数设置
#！！！特别注意：这里使用了 torchcodec 解码器，而不是 pyav 解码器。
dataset = LeRobotDataset(
    repo_id="lerobot/pusht",
    revision="v3.0",
    video_backend="torchcodec",
    delta_timestamps={
        "observation.image": obs_times,
        "observation.state": obs_times,
        "action": act_times
    }
)

#PyTorch 数据加载器，它把 dataset 切成 batch，每个 batch 包含 2 个样本。
#shuffle=False 表示不打乱样本顺序，num_workers=0 表示在主进程中加载数据。
#shuffle=True 可以在每个 epoch 重新打乱样本顺序，num_workers>0 可以使用多个子进程加载数据。

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
# for key in ("observation.image", "observation.state", "action"):
#     print(key, tuple(batch[key].shape))
# observation.image (2, 2, 3, 96, 96)                                      |  0.00B /  0.00B            
# observation.state (2, 2, 2)
# action (2, 64, 2)
print("样本数", len(dataset))
print("每轮批次数", len(loader))


#cfg 保存 Diffusion Policy 的配置。
cfg = DiffusionConfig(
    device="cuda",
    n_obs_steps=2,#每个样本输入两帧观察，每帧包含画面和小球位置。
    horizon=horizon,
    n_action_steps=8,
    down_dims=(64, 128, 256),#	一维 U-Net 的三个阶段分别使用 64、128、256 个特征通道。这些数字决定网络每个阶段的宽度。
                            #LeRobot 在前两级末尾用了 stride=2 的一维卷积。卷积窗口每次向前移动 2 个时间位置，每个窗口算出 1 个输出位置。
    diffusion_step_embed_dim=32,#把扩散时间步编号编码成一个包含 32 个数的向量，告诉模型当前的加噪程度。
)
#根据 cfg 创建模型，从数据集元数据读取图像、状态和动作的形状。
policy = make_policy(cfg, ds_meta=dataset.meta) 
#把模型切换到训练模式。
policy.train()

#创建两个处理器
# preprocessor 在数据送进模型前使用，动作后处理器在模型生成动作后使用。
# - cfg 决定处理规则，比如用哪种归一化方式、把张量放到哪个设备。
# - dataset.meta.stats 提供归一化需要的数字，比如图像的均值、标准差，以及小球位置和动作坐标的最小值、最大值。
preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.optimizer_lr)

#训练200步，记录每步的 loss。
#每步从数据集 loader 中取一个 batch，经过预处理器处理后送入模型计算 loss 并更新参数。
max_steps = 1000
# save_steps = {2000, 5000, 10000}

step = 0
loss_history = []

run_name = f"train_{datetime.now():%Y%m%d_%H%M%S_%f}"
output_dir = Path(__file__).resolve().parent / "outputs" / run_name
output_dir.mkdir(parents=True)

while step < max_steps:
    for batch in loader:
        batch = preprocessor(batch)
        optimizer.zero_grad()
        loss, _ = policy(batch)
        loss.backward()
        optimizer.step()

        step += 1
        loss_history.append(loss.item())
        print(f"Step {step:04d}/{max_steps}, Loss: {loss.item():.6f}")
        # if step in save_steps:
        #     model_dir = output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
        #     model_dir.mkdir(parents=True)

        #     policy.save_pretrained(model_dir)
        #     preprocessor.save_pretrained(model_dir)
        #     postprocessor.save_pretrained(model_dir)

        if step == max_steps:
            break

print(f"完成 {step} 次参数更新")
print(f"最后一次 loss = {loss_history[-1]:.6f}")


#保存原始loss数据到 outputs/train_YYYYMMDD_HHMMSS_ffffff 目录下。
steps = range(1, len(loss_history) + 1)



# 保存每一步的原始 loss
with (output_dir / "loss.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["step", "loss"])
    writer.writerows(zip(steps, loss_history))


#绘制 loss 曲线并保存为 PNG 文件
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, loss_history, linewidth=1.2)
ax.set(xlabel="Training step", ylabel="Loss", title="PushT Diffusion Policy")
ax.grid(alpha=0.25)
fig.tight_layout()

plot_path = output_dir / "loss_curve.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)

print(f"Loss 曲线保存到 {plot_path}")