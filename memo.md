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


Day2

python: can't open file '/home/ito-marl/ur3e_maniskill/scripts/day2_test_pick_lift_env.py': [Errno 2] No such file or directory
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/test_pick_lift_env.py \
  --sim-backend physx_cpu \
  --steps 1000 \
  --render \
  --sleep 0.01
========================================================================
Day 2 UR3ePickLift-v0 smoke test
========================================================================
env id       : UR3ePickLift-v0
robot uid    : ur3e_ezgripper
control mode : pd_joint_delta_pos
sim backend  : physx_cpu
obs shape    : torch.Size([1, 34])
action shape : (7,)
action low   : [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
action high  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
joint names  : ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'gripper_ezgripper_knuckle_palm_L1_1', 'gripper_ezgripper_knuckle_palm_L1_2']
link names   : ['world', 'base_link', 'base_link_inertia', 'base', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'ft_frame', 'flange', 'tool0', 'gripper_ezgripper_mount', 'gripper_ezgripper_palm_link', 'gripper_ezgripper_finger_L1_1', 'gripper_ezgripper_finger_L1_2', 'grasp_tcp', 'gripper_ezgripper_finger_L2_1', 'gripper_ezgripper_finger_L2_2', 'gripper_ezgripper_finger_pad_1', 'gripper_ezgripper_finger_pad_2']
initial qpos : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
cube initial : [0.4000000059604645, 0.0, 0.02500000037252903]
step=   0, cube=[0.39999982714653015, -7.05397212641401e-07, 0.02499990537762642], reward=0.032143
step=  50, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 100, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 150, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 200, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 250, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 300, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 350, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 400, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 450, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 500, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 550, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 600, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 650, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 700, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 750, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 800, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 850, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 900, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 950, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
------------------------------------------------------------------------
final cube   : [0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733]
cube XY drift: 0.000001 m
cube Z error : 0.000000 m
max q drift  : 6.503711e-01 rad
report       : reports/day2_pick_lift_env.json
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/test_pick_lift_env.py   --sim-backend physx_cpu   --steps 1000   --render   --sleep 0.01
========================================================================
Day 2 UR3ePickLift-v0 smoke test
========================================================================
env id       : UR3ePickLift-v0
robot uid    : ur3e_ezgripper
control mode : pd_joint_delta_pos
sim backend  : physx_cpu
obs shape    : torch.Size([1, 34])
action shape : (7,)
action low   : [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
action high  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
joint names  : ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'gripper_ezgripper_knuckle_palm_L1_1', 'gripper_ezgripper_knuckle_palm_L1_2']
link names   : ['world', 'base_link', 'base_link_inertia', 'base', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'ft_frame', 'flange', 'tool0', 'gripper_ezgripper_mount', 'gripper_ezgripper_palm_link', 'gripper_ezgripper_finger_L1_1', 'gripper_ezgripper_finger_L1_2', 'grasp_tcp', 'gripper_ezgripper_finger_L2_1', 'gripper_ezgripper_finger_L2_2', 'gripper_ezgripper_finger_pad_1', 'gripper_ezgripper_finger_pad_2']
initial qpos : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
cube initial : [0.4000000059604645, 0.0, 0.02500000037252903]
step=   0, cube=[0.39999982714653015, -7.05397212641401e-07, 0.02499990537762642], reward=0.032143
step=  50, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 100, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 150, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 200, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 250, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 300, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 350, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 400, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 450, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 500, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 550, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 600, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 650, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 700, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 750, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 800, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 850, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 900, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
step= 950, cube=[0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733], reward=0.026242
------------------------------------------------------------------------
final cube   : [0.39999979734420776, -6.972633173063514e-07, 0.024999871850013733]
cube XY drift: 0.000001 m
cube Z error : 0.000000 m
max q drift  : 6.503711e-01 rad
report       : reports/day2_pick_lift_env.json






(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python -m py_compile   scripts/day3_inspect_gripper_controller.py
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day3_inspect_gripper_controller.py \
  --env-id UR3ePickLift-v0 \
  --sim-backend physx_cpu
========================================================================
Day 3 Controller Inspection
========================================================================
environment action shape: (7,)
active joints: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'gripper_ezgripper_knuckle_palm_L1_1', 'gripper_ezgripper_knuckle_palm_L1_2']

controller : arm
class      : PDJointPosController
config     : PDJointPosControllerConfig
joints     : ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
slice      : [0, 6]
space      : {'shape': [6], 'low': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], 'high': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
------------------------------------------------------------------------
controller : gripper
class      : PDJointPosMimicController
config     : PDJointPosMimicControllerConfig
joints     : ['gripper_ezgripper_knuckle_palm_L1_1', 'gripper_ezgripper_knuckle_palm_L1_2']
slice      : [6, 7]
space      : {'shape': [1], 'low': [-1.0], 'high': [1.0]}
------------------------------------------------------------------------
report: reports/day3_gripper_controller.json
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python -m json.tool \
  reports/day3_gripper_controller.json
{
    "env_id": "UR3ePickLift-v0",
    "robot_uid": "ur3e_ezgripper",
    "control_mode": "pd_joint_delta_pos",
    "sim_backend": "physx_cpu",
    "observation_shape": [
        1,
        34
    ],
    "environment_action_space": {
        "shape": [
            7
        ],
        "low": [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0
        ],
        "high": [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ]
    },
    "active_joint_names": [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        "gripper_ezgripper_knuckle_palm_L1_1",
        "gripper_ezgripper_knuckle_palm_L1_2"
    ],
    "qpos_initial": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ],
    "qlimits": [
        [
            -6.2831854820251465,
            6.2831854820251465
        ],
        [
            -6.2831854820251465,
            6.2831854820251465
        ],
        [
            -3.1415927410125732,
            3.1415927410125732
        ],
        [
            -6.2831854820251465,
            6.2831854820251465
        ],
        [
            -6.2831854820251465,
            6.2831854820251465
        ],
        [
            -Infinity,
            Infinity
        ],
        [
            -1.5707499980926514,
            0.27000001072883606
        ],
        [
            -1.5707499980926514,
            0.27000001072883606
        ]
    ],
    "controllers": [
        {
            "name": "arm",
            "class": "PDJointPosController",
            "config_class": "PDJointPosControllerConfig",
            "joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint"
            ],
            "action_slice": [
                0,
                6
            ],
            "action_space": {
                "shape": [
                    6
                ],
                "low": [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0
                ],
                "high": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0
                ]
            }
        },
        {
            "name": "gripper",
            "class": "PDJointPosMimicController",
            "config_class": "PDJointPosMimicControllerConfig",
            "joint_names": [
                "gripper_ezgripper_knuckle_palm_L1_1",
                "gripper_ezgripper_knuckle_palm_L1_2"
            ],
            "action_slice": [
                6,
                7
            ],
            "action_space": {
                "shape": [
                    1
                ],
                "low": [
                    -1.0
                ],
                "high": [
                    1.0
                ]
            }
        }
    ],
    "warnings": [],
    "errors": []
}


(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python -m py_compile \
  scripts/day3_test_gripper_cycles.py
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day3_test_gripper_cycles.py \
  --env-id UR3ePickLift-v0 \
  --sim-backend physx_cpu \
  --cycles 3 \
  --phase-steps 30 \
  --render \
  --sleep 0.01
========================================================================
Day 3 EZGripper cycle test
========================================================================
cycles              : 3
phase steps         : 30
gripper controller  : gripper
gripper slice       : [6, 7]
gripper joints      : ['gripper_ezgripper_knuckle_palm_L1_1', 'gripper_ezgripper_knuckle_palm_L1_2']
gripper indices     : [6, 7]
open action         : 1.0
close action        : -1.0
cycle=  1/3, max|qvel|=0.005960
cycle=  3/3, max|qvel|=0.000075
------------------------------------------------------------------------
completed cycles     : 3
max gripper motion   : 1.840550 rad
max arm drift        : 0.135739 rad
max open std         : 0.000000 rad
max close std        : 0.000268 rad
[WARN] 腕の関節ドリフトが暫定上限を超えました: 0.135739 rad > 0.010000 rad
report: reports/day3_gripper_cycles.json

