from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
from torch.utils.data import DataLoader

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

import csv
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt

#一、dataset:采用LeRobotDataset加载pusht数据集

#pusht数据集的演示视频帧率为 10 fps。
#每个训练样本包含两帧观察，以及从上一帧开始、长度为 16 的动作片段。
fps=10
horizon=16  #一次生成长度为 16 的动作序列
first_offset=-1 #从当前帧的上一帧开始

# 计算每个样本的 观察帧 和 动作帧 的时间戳

#每次预测输入上一帧和当前帧
obs_times=[first_offset/fps,0.0]

#每次预测输出从上一帧开始的连续 16 帧动作
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

#二、DataLoader
#PyTorch 数据加载器，它把 dataset 切成 batch，每个完整 batch 包含 64 个样本。
#shuffle=False 表示不打乱样本顺序
#shuffle=True  可以在每个 epoch 重新打乱样本顺序，
#num_workers>0 可以使用多个子进程加载数据。

loader = DataLoader(
    dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=2
    )
print("样本数", len(dataset))
print("每轮批次数", len(loader))


#三、cfg 保存 Diffusion Policy 的配置。
cfg = DiffusionConfig(
    device="cuda",
    n_obs_steps=2,#每个样本输入两帧观察，每帧包含画面和小球位置。
    horizon=horizon, #每次预测输出从上一帧开始的连续 16 帧动作
    n_action_steps=8,#每次执行8帧动作后重新预测
    down_dims=(512, 1024, 2048),#一维 U-Net 的三个阶段分别使用 512、1024、2048 个特征通道。这些数字决定网络每个阶段的宽度。
                             #LeRobot 在前两级末尾用了 stride=2 的一维卷积。卷积窗口每次向前移动 2 个时间位置，每个窗口算出 1 个输出位置。
    diffusion_step_embed_dim=32,#把扩散时间步编号编码成一个包含 32 个数的向量，告诉模型当前的加噪程度。
)
#根据 cfg 创建模型，从数据集元数据读取图像、状态和动作的形状。
policy = make_policy(cfg, ds_meta=dataset.meta) 
#把模型切换到训练模式。
policy.train()

#四、创建两个处理器
# preprocessor 在数据送进模型前使用，动作后处理器在模型生成动作后使用。
# - cfg 决定处理规则，比如用哪种归一化方式、把张量放到哪个设备。
# - dataset.meta.stats 提供归一化需要的数字，比如图像的均值、标准差，以及小球位置和动作坐标的最小值、最大值。
preprocessor, postprocessor = make_pre_post_processors(
    cfg, 
    dataset_stats=dataset.meta.stats
    )

optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.optimizer_lr)


# 五、训练Diffusion Policy参数
#最大训练步数
max_steps = 175000
#在指定训练步数处保存模型和处理器
save_steps = {55000, 110000, 175000}

step = 0
loss_history = []

run_name = f"train_{datetime.now():%Y%m%d_%H%M%S_%f}"
output_dir = Path(__file__).resolve().parent / "outputs" / run_name
output_dir.mkdir(parents=True)

print(f"输出目录：{output_dir}", flush=True)
print(
    f"训练配置：总步数={max_steps}，"
    f"batch_size={loader.batch_size}，"
    f"num_workers={loader.num_workers}",
    flush=True,
)
print(
    f"模型配置：观察={cfg.n_obs_steps}，"
    f"预测={cfg.horizon}，执行={cfg.n_action_steps}，"
    f"U-Net通道={cfg.down_dims}",
    flush=True,
)
print(f"保存检查点的步数：{sorted(save_steps)}", flush=True)

progress_path = output_dir / "progress.csv"

with progress_path.open("w", newline="", encoding="utf-8") as file:
    csv.writer(file).writerow([
        "step",
        "loss",
        "avg_loss",
        "seconds_per_step",
        "samples_per_second",
        "remaining_seconds",
    ])



#用于记录每步训练的时间和样本数
timing_rows = []

torch.cuda.synchronize()
step_start = perf_counter()

while step < max_steps:
    for batch in loader:
        batch = preprocessor(batch)#加载数据后，先经过预处理器处理，再送入模型计算 loss 并更新参数。
        optimizer.zero_grad()      #清掉上一批留下的梯度
        loss, _ = policy(batch)    #用当前参数计算这一批的损失
        loss.backward()            #计算损失对模型参数的梯度
        optimizer.step()           #Adam 根据梯度更新模型参数

        
        # 等待本步 GPU 运算完成，再结束计时
        torch.cuda.synchronize()
        step_seconds = perf_counter() - step_start

        step += 1
        loss_value = loss.item()
        loss_history.append(loss_value)

        # 最后一批可能不足 64 个，记录实际样本数
        actual_batch_size = batch["action"].shape[0]
        timing_rows.append((step, step_seconds, actual_batch_size))

        # 每 50 步打印并保存一次进度，减少终端输出
        if len(timing_rows) % 50 == 0 or step == max_steps:
            recent = timing_rows[-50:]
            recent_count = len(recent)
            total_seconds = sum(row[1] for row in recent)
            total_samples = sum(row[2] for row in recent)

            avg_step_seconds = total_seconds / recent_count
            samples_per_second = total_samples / total_seconds
            avg_loss = sum(loss_history[-recent_count:]) / recent_count

            # 按最近的训练速度估算，计时不包含日志输出和检查点保存耗时。
            remaining_seconds = avg_step_seconds * (max_steps - step)
            remaining_time = timedelta(seconds=round(remaining_seconds))

            print(
                f"Step {step}/{max_steps} "
                f"({step / max_steps:.1%}) | "
                f"loss={loss_value:.5f} | "
                f"最近{recent_count}步平均loss={avg_loss:.5f} | "
                f"{avg_step_seconds:.3f} s/step | "
                f"{samples_per_second:.1f} samples/s | "
                f"预计剩余={remaining_time}",
                flush=True,
            )

            with progress_path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([
                    step,
                    loss_value,
                    avg_loss,
                    avg_step_seconds,
                    samples_per_second,
                    remaining_seconds,
                ])

        # 在指定训练步数处保存模型和处理器
        if step in save_steps:
            model_dir = output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
            model_dir.mkdir(parents=True)

            print(f"开始保存第 {step} 步检查点……", flush=True)
            save_start = perf_counter()

            policy.save_pretrained(model_dir)
            preprocessor.save_pretrained(model_dir)
            postprocessor.save_pretrained(model_dir)

            save_seconds = perf_counter() - save_start
            print(
                f"检查点保存完成，用时 {save_seconds:.1f} 秒\n"
                f"保存位置：{model_dir}",
                flush=True,
            )

        step_start = perf_counter()

        if step == max_steps:
            break

print(f"完成 {step} 次参数更新")
print(f"最后一次 loss = {loss_history[-1]:.6f}")
print(f"训练进度记录保存到 {progress_path}")


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
