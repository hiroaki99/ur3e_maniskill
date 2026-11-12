# ur3e_maniskill

UR3e + EZGripper を ManiSkill 3 上で動かし、Residual RL（TD3）と Yaginuma の Expert GP 軌道を組み合わせて Pick-and-Lift / Pick-and-Place / Alignment タスクを学習・評価するプロジェクト。sim2real 用に、ROS 2 経由で実機の UR3e を動かすバックエンドも含む。

## 動作確認済み環境

| 項目 | バージョン |
| --- | --- |
| OS | Ubuntu 24.04 LTS |
| GPU | NVIDIA GeForce RTX 4070 Laptop（ドライバ 595） |
| Python（シミュレーション・学習用） | 3.11（conda 環境 `maniskill_yaginuma`） |
| ManiSkill / SAPIEN | 3.0.1 / 3.0.3 |
| PyTorch | 2.13.0（CUDA 13.0） |
| ROS 2（実機用のみ） | Jazzy（システムの `/usr/bin/python3` 3.12 を使う） |

## ディレクトリ構成

```
agents/            UR3e / UR3e+EZGripper の ManiSkill エージェント定義
assets/            URDF・メッシュ
configs/           タスク設定（pick_lift / pick_place / alignment）
envs/              ManiSkill 環境（reach, pick_lift, pick_place, alignment）
rrl/               Residual RL 本体（TD3, 報酬, 観測スケーリングなど）
expert_gp/         Yaginuma の Expert GP（GaussianProcess.pyx, data/class000〜011.npy）
scripts/           Pick-and-Lift の開発・学習・評価スクリプト（day*）
scripts_pickplace/ Pick-and-Place の開発・学習・評価スクリプト
scripts_expert/    Expert GP の生成・評価、Alignment / Phase6 の実験スクリプト
scripts_sim2real/  実機 UR3e 用 ROS 2 ブリッジと制御バックエンド
memo.md            作業メモ
```

## 1. システムパッケージ

```bash
sudo apt update
sudo apt install -y git build-essential libvulkan1 vulkan-tools
```

- `build-essential`：`expert_gp/GaussianProcess.pyx` を Cython でビルドするのに必要。
- `libvulkan1`：SAPIEN のレンダリングに必要。

先に NVIDIA ドライバを入れておく（「ソフトウェアとアップデート」→「追加のドライバー」、または `sudo ubuntu-drivers install`）。動作確認は次のとおり。

```bash
nvidia-smi
vulkaninfo --summary   # GPU が表示されれば OK
```

## 2. Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# シェルを開き直す
```

## 3. リポジトリの取得

```bash
cd ~/デスクトップ
git clone https://github.com/hiroaki99/ur3e_maniskill.git
cd ur3e_maniskill
git checkout test      # 作業ブランチ
```

## 4. Python 環境（シミュレーション・学習用）

```bash
conda create -n maniskill_yaginuma python=3.11 -y
conda activate maniskill_yaginuma

pip install -r requirements.txt
```

`requirements.txt` には現在の環境のパッケージをすべてバージョン固定で記載している（torch 2.13.0 + CUDA 13.0、mani_skill 3.0.1 など）。バージョンを固定せずに `pip install torch mani_skill` などで入れると、より新しい版（torch 2.15、gymnasium 1.4 など）が入って環境が一致しなくなるため、必ずこのファイルを使う。

動作確認：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import mani_skill, sapien; print(mani_skill.__version__, sapien.__version__)"
python -m mani_skill.examples.demo_random_action   # 最後に info {'elapsed_steps': ...} が表示されれば OK
```

## 5. Expert GP（Cython）のビルド確認

`expert_gp/GaussianProcessMultiDim.py` は `pyximport` を使っており、初回の import 時に `GaussianProcess.pyx` が自動でビルドされ、`expert_gp/` に `.so` が作られる。`.so` のファイル名には Python のバージョンが入る（例：`cpython-311`）ので、バージョンが違う環境では自動で別の `.so` がビルドされる。

```bash
python scripts_expert/generate_expert_gps.py   # 12 本の Expert 軌道を reports/expert_gp/ に生成
python scripts_expert/plot_expert_gps.py       # 軌道図を出力（任意）
```

`reports/expert_gp/` は `.gitignore` に入っているため、新しい環境では最初にこのコマンドで作り直す。

## 6. 動作確認・主な実行例

スクリプトはリポジトリのルートから実行する（`envs`, `rrl`, `agents` を import するため）。

```bash
conda activate maniskill_yaginuma
cd ~/デスクトップ/ur3e_maniskill

# Pick-and-Lift 環境
python scripts/test_pick_lift_env.py

# 12 Expert の ManiSkill 上での一括評価
python scripts_expert/evaluate_all_experts_maniskill.py

# Alignment：yaw-aware 把持（Phase 5B2 で PASS した版）
python scripts_expert/test_alignment_grasp_A_yaw_aware.py
python scripts_expert/evaluate_alignment_grasp_multiseed_yaw_aware.py

# Alignment：Expert06 / 643 系列の strict 版、Phase6 の residual 注入
python scripts_expert/test_alignment_grasp_A_fixed_643_strict.py
python scripts_expert/test_phase6_residual_injection.py
```

それぞれの引数は `--help` で確認できる。`evaluate_all_experts_maniskill.py` は FAIL の Expert（00, 01, 07, 08, 10）があると終了コード 2 で終わるが、これは現状の想定どおりの結果。

学習結果（`runs/`, `checkpoints/`, `*.pt` など）は git に入らないため、必要なら別途バックアップしておく。

### Pick-and-Lift / Pick-and-Place の参照軌道

Residual RL の学習・評価スクリプト（`scripts/day13_*`, `scripts_pickplace/` など）は `trajectories/*.json` の参照軌道を読み込む。`*.json` は git に入らないため、新しい環境では作り直す必要がある（`memo.md` に当時の実行コマンドが残っている）。

- Pick-and-Lift：`scripts/day6_plan_reference.py`（出力の既定値は `trajectories/day6_pick_lift_reference.json`）
- Pick-and-Place：`scripts_pickplace/pickplace_day04_plan_reference.py`（入力に `trajectories/day6_pick_lift_reference_v2.json` を使う）

`_v2` を作ったときの引数は記録が残っていないため、この 2 つのスクリプトでの再生成手順はまだ確認していない。

## 7. 実機 UR3e（sim2real、必要なときだけ）

### 7.1 ROS 2 Jazzy と UR ドライバ

[ROS 2 Jazzy のインストール手順](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)に従って `ros-jazzy-desktop` を入れたあと、次を入れる。

```bash
sudo apt install -y ros-jazzy-ur ros-jazzy-ur-robot-driver ros-jazzy-xacro
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
```

### 7.2 起動手順

PC 側のネットワークを UR3e と同じサブネットに設定する（UR3e の IP：`192.168.2.4`）。ROS 2 のコマンドとブリッジは conda 環境の Python ではなく、システムの Python（3.12）で動かす。`~/.bashrc` で ROS 2 を source すると、conda 環境でも `PYTHONPATH` に ROS 2 のパスが入るが、シミュレーション側の動作には影響しないことを確認済み。

```bash
# ターミナル 1：UR ドライバ
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.2.4 launch_rviz:=false

# ターミナル 2：ブリッジ（システムの Python で実行する）
cd ~/デスクトップ/ur3e_maniskill
/usr/bin/python3 scripts_sim2real/ur3e_ros2_bridge.py
```

ブリッジは TCP ポート `50051` で待ち受ける。conda 環境側の `scripts_sim2real/ur3e_ros2_backend.py` からこのポートに接続して実機を動かす。ティーチペンダントで External Control プログラムを実行しておくこと。

## 注意事項

- `third_party/EZGripper_ros2` はサブモジュールとして登録されているが `.gitmodules` が無いため、clone 後は空のフォルダになる。シミュレーションで使う URDF とメッシュは `assets/` に入っているので、シミュレーションには不要。
- `memo.md` の古いログには、以前の PC のパス（`~/ur3e_maniskill`, conda 環境 `maniskill`）が残っている。
