(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/diagnose_ur3e_reach.py

[Zero action]
initial distance mean : 0.0222
minimum distance mean : 0.0222
final distance mean   : 0.0222
return mean           : 92.7734
success once          : 16/16

[Random action]
initial distance mean : 0.0222
minimum distance mean : 0.0145
final distance mean   : 0.0756
return mean           : 48.9572
success once          : 16/16


作業メモ
Day1
作成するファイル
ur3e_maniskill/
├── configs/
│   └── ur3e_pick_lift.yaml
│
├── scripts/
│   ├── validate_pick_lift_config.py
│   ├── inspect_ur3e.py
│   └── run_day1_checks.sh
│
└── reports/
    ├── day1_pick_lift_spec.json
    ├── day1_pick_lift_spec.md
    └── day1_ur3e_inspection.json

(base) ito-marl@ubuntu:~$ conda activate maniskill
(maniskill) ito-marl@ubuntu:~$ chmod +x ~/ur3e_maniskill/scripts/run_day1_checks.sh 

(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/validate_pick_lift_config.py   --config configs/ur3e_pick_lift.yaml   --output-dir reports
========================================================================
Day 1 Pick-and-Lift仕様検証
========================================================================
grasp TCP   : [0.4, 0.0, 0.025]
pregrasp TCP: [0.4, 0.0, 0.125]
lift TCP    : [0.4, 0.0, 0.07500000000000001]
実効残差上限: 0.006 rad/step
JSON: reports/day1_pick_lift_spec.json
Markdown: reports/day1_pick_lift_spec.md


(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day1_inspect_ur3e.py   --config configs/ur3e_pick_lift.yaml   --env-id UR3eReach-v0   --register-module envs.ur3e_reach   --steps 20
========================================================================
Day 1 UR3e実環境検査
========================================================================
env_id: UR3eReach-v0
joint names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
qpos: [0.0, -1.5707963705062866, 1.5707963705062866, -1.5707963705062866, -1.5707963705062866, 0.0]
action shape: [6]
action low: [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
action high: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
obs shape: [1, 25]
robot base: [0.0, 0.0, 0.0]
initial TCP: [0.2985500693321228, 0.13104994595050812, 0.3032999634742737]
grasp target: [0.4, 0.0, 0.025]
pregrasp: [0.4, 0.0, 0.125]
lift target: [0.4, 0.0, 0.07500000000000001]
TCP→grasp: 0.3239 m
最大qpos drift: 0.000000e+00 rad
report: reports/day1_ur3e_inspection.json







