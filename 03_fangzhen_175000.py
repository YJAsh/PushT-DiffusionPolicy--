import subprocess
import sys
from pathlib import Path

run_dir = Path("/home/lucyan/pusht_DiffusionPolicy/outputs/train_20261117_093517_401881")
for step in (55000, 110000):
    name = f"{step:06d}"
    model_dir = run_dir / "checkpoints" / name / "pretrained_model"
    eval_dir = run_dir / "eval" / name

    subprocess.run(
        [
            sys.executable, "-m", "lerobot.scripts.lerobot_eval",
            f"--policy.path={model_dir}",
            "--env.type=pusht",
            "--env.episode_length=500",
            "--seed=2026",
            "--eval.n_episodes=10",
            "--eval.batch_size=1",
            f"--output_dir={eval_dir}",
        ],
        check=True,
    )