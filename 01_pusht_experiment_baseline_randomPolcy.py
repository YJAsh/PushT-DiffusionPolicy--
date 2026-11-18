import gymnasium as gym 
import gym_pusht
import numpy as np

import json
from datetime import datetime
from pathlib import Path
from importlib.metadata import version

env=gym.make(
    "gym_pusht/PushT-v0", 
    obs_type="state",       # 返回位置、角度等状态数据
    render_mode="human",
    max_episode_steps=300
    )

n_episodes = 10
n_success = 0
max_rewards = []

# 记录列表
episode_results = []

try:
    for episode in range(n_episodes):
        #共训练10轮，每一轮的随机数种子设定为2026+episode
        seed = 2026 + episode
        
        #重置环境，使用指定随机种子生成可复现的初始状态
        #默认情况下，每局变化的是小球和 T 块的起始状态，绿色目标区域保持固定
        #每局变化的是observation,observation类型指定为“state”,则包含[小球x坐标, 小球y坐标, T块x坐标, T块y坐标, T块角度]
        #绿色目标区域则使用固定的位置和角度：goal_pose = [256, 256, π / 4]  # 目标位置和 45° 朝向
        observation, info = env.reset(seed=seed)

        #给动作空间的随机数生成器设置种子，使得后面的随机动作可以复现。
        env.action_space.seed(seed)

        # print(f"第{episode+1}局，初始小球位置{observation[:2]}")

        best_reward = 0.0

        steps = 0

        while True:
            action = env.action_space.sample()
            steps+=1
            observation, reward, terminated, truncated, info = env.step(action)
            # 奖励公式：reward = np.clip(coverage / 0.95, 0.0, 1.0)
            # terminated	布尔值 True/False	是否达到任务本身的结束条件
            # truncated	    布尔值 True/False	是否因为步数上限等外部限制而结束
            # info	        字典 dict	        额外信息，例如是否成功、重合比例

            env.render()

            best_reward = max(best_reward, reward)

            if terminated or truncated:
                break

     
        # 记录每局是否成功
        success = info["is_success"]
        n_success += int(success)

        # 记录每局的最高奖励
        max_rewards.append(best_reward)
        # 记录每局的结果
        episode_results.append({
                            "episode": episode + 1,
                            "env_seed": seed,
                            "action_seed": seed,
                            "success": bool(success),
                            "max_reward": float(best_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        })
        # 打印每局的结果
        # print(f"成功 {success}，本局最高奖励 {best_reward:.3f}")
finally:
    env.close()

avg_max_reward = sum(max_rewards) / len(max_rewards)
print(f"成功次数 {n_success}/{n_episodes}")
print(f"平均最高奖励 {avg_max_reward:.4f}")

# 保存结果到 JSON 文件
results = {
    "config": {
        "policy": "random",
        "env_id": env.spec.id,
        "max_episode_steps": env.spec.max_episode_steps,
        "n_episodes": n_episodes,
        "versions": {
            name: version(name)
            for name in ("gymnasium", "gym-pusht", "numpy", "pymunk")
        },
    },
    "summary": {
        "n_episodes": len(episode_results),
        "n_success": n_success,
        "pc_success": n_success / len(episode_results) * 100,
        "avg_max_reward": float(avg_max_reward),
    },
    "episodes": episode_results,
}

# 以当前 Python 文件所在文件夹作为项目目录
project_dir = Path(__file__).resolve().parent

# 每次运行创建一个带时间戳的新目录
run_name = f"random_baseline_{datetime.now():%Y%m%d_%H%M%S_%f}"
output_dir = project_dir / "outputs" / run_name
#创建输出目录，如果目录已存在则抛出异常
output_dir.mkdir(parents=True, exist_ok=False)

# 将结果保存到 JSON 文件
result_file = output_dir / "results.json"
result_file.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("基线结果已保存到：", result_file)