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


(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_sweep_contact_position.py 
calculated center: [0.29854998 0.13105005 0.17586371]
/home/ito-marl/ur3e_maniskill/scripts/day4_sweep_contact_position.py:181: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  left_force = float(
/home/ito-marl/ur3e_maniskill/scripts/day4_sweep_contact_position.py:187: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  right_force = float(
dx=-0.150 position=[0.14854997 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.140 position=[0.15854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.130 position=[0.16854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.120 position=[0.17854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.110 position=[0.18854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.100 position=[0.19854999 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.090 position=[0.20854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.080 position=[0.21854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.070 position=[0.22854999 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.060 position=[0.23854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.050 position=[0.24854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.040 position=[0.25855    0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.030 position=[0.26854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.020 position=[0.27854997 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=-0.010 position=[0.28855    0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.000 position=[0.29854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.010 position=[0.30854997 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.020 position=[0.31855    0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.030 position=[0.32854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.040 position=[0.33854997 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.050 position=[0.34855    0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.060 position=[0.35854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.070 position=[0.36854997 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.080 position=[0.37855    0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.090 position=[0.38854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.100 position=[0.39854997 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.110 position=[0.40854996 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.120 position=[0.41854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.130 position=[0.42854998 0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.140 position=[0.43855    0.13105005 0.17586371] L=0.00000 N R=0.00000 N
dx=+0.150 position=[0.44855    0.13105005 0.17586371] L=0.00000 N R=0.00000 N
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python -m py_compile \
  scripts/day4_diagnose_zero_contact.py
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_diagnose_zero_contact.py
========================================================================
1. Cube - Table Contact API Sanity Check
========================================================================
cube-table force vector: [[0.        0.        0.9809999]]
cube-table force norm: 0.9809998869895935 N
[OK] pairwise contact APIは機能している可能性が高いです。

========================================================================
2. Finger Collision Geometry
========================================================================

------------------------------------------------------------------------
link: gripper_ezgripper_finger_pad_1
collision shape count: 1
  shape[0] type=PhysxCollisionShapeConvexMesh groups=[1, 1, 0, 0]
collision AABB min: [0.15684463 0.11608958 0.23271557]
collision AABB max: [0.20572222 0.14632895 0.25037133]
collision center  : [0.18128342 0.13120926 0.24154345]
collision extent  : [0.04887759 0.03023936 0.01765576]

------------------------------------------------------------------------
link: gripper_ezgripper_finger_pad_2
collision shape count: 1
  shape[0] type=PhysxCollisionShapeConvexMesh groups=[1, 1, 0, 0]
collision AABB min: [0.39149496 0.11577333 0.23279242]
collision AABB max: [0.44036427 0.1460124  0.25045603]
collision center  : [0.41592961 0.13089287 0.24162423]
collision extent  : [0.04886931 0.03023907 0.01766362]

========================================================================
3. Collision-based Grasp Center
========================================================================
left collision center : [0.18128342 0.13120926 0.24154345]
right collision center: [0.41592961 0.13089287 0.24162423]
grasp center candidate: [0.29860652 0.13105107 0.24158384]
collision center gap   : 0.2346464175377155 m

old link-origin center : [0.2986047  0.13105114 0.2438209 ]
center difference      : [ 1.82012325e-06 -7.21053716e-08 -2.23706929e-03]
difference norm        : 0.002237070031899353 m
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_diagnose_zero_contact.py
========================================================================
1. Cube - Table Contact API Sanity Check
========================================================================
cube-table force vector: [[0.        0.        0.9809999]]
cube-table force norm: 0.9809998869895935 N
[OK] pairwise contact APIは機能している可能性が高いです。

========================================================================
2. Finger Collision Geometry
========================================================================

------------------------------------------------------------------------
link: gripper_ezgripper_finger_pad_1
collision shape count: 1
  shape[0] type=PhysxCollisionShapeConvexMesh groups=[1, 1, 0, 0]
collision AABB min: [0.15684463 0.11608958 0.23271557]
collision AABB max: [0.20572222 0.14632895 0.25037133]
collision center  : [0.18128342 0.13120926 0.24154345]
collision extent  : [0.04887759 0.03023936 0.01765576]

------------------------------------------------------------------------
link: gripper_ezgripper_finger_pad_2
collision shape count: 1
  shape[0] type=PhysxCollisionShapeConvexMesh groups=[1, 1, 0, 0]
collision AABB min: [0.39149496 0.11577333 0.23279242]
collision AABB max: [0.44036427 0.1460124  0.25045603]
collision center  : [0.41592961 0.13089287 0.24162423]
collision extent  : [0.04886931 0.03023907 0.01766362]

========================================================================
3. Collision-based Grasp Center
========================================================================
left collision center : [0.18128342 0.13120926 0.24154345]
right collision center: [0.41592961 0.13089287 0.24162423]
grasp center candidate: [0.29860652 0.13105107 0.24158384]
collision center gap   : 0.2346464175377155 m

old link-origin center : [0.2986047  0.13105114 0.2438209 ]
center difference      : [ 1.82012325e-06 -7.21053716e-08 -2.23706929e-03]
difference norm        : 0.002237070031899353 m
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python -m py_compile \
  scripts/day4_measure_gripper_gap.py
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_measure_gripper_gap.py

========================================================================
OPEN
========================================================================
left collision center : [0.18122648 0.13120792 0.24155959]
right collision center: [0.41587275 0.13089153 0.24156152]
grasp center          : [0.29854961 0.13104973 0.24156055]
center distance       : 0.234646 m
inner gap X           : 0.185773 m
cube size             : 0.050000 m
gap - cube size       : 0.135773 m
[INFO] この姿勢ではcubeより135.8 mm広いです。

========================================================================
PRE-CONTACT
========================================================================
left collision center : [0.27351526 0.13120838 0.15671791]
right collision center: [0.32358511 0.13089206 0.15671752]
grasp center          : [0.29855019 0.13105022 0.15671772]
center distance       : 0.050071 m
inner gap X           : 0.032807 m
cube size             : 0.050000 m
gap - cube size       : -0.017193 m
[OK] この姿勢では50 mm cubeに接触可能な間隔です。

========================================================================
CLOSE
========================================================================
left collision center : [0.28977195 0.13120826 0.16084651]
right collision center: [0.30732802 0.13089195 0.16084649]
grasp center          : [0.29854998 0.1310501  0.1608465 ]
center distance       : 0.017559 m
inner gap X           : 0.000999 m
cube size             : 0.050000 m
gap - cube size       : -0.049001 m
[OK] この姿勢では50 mm cubeに接触可能な間隔です。

========================================================================
Summary
========================================================================
OPEN gap: 0.1857728087001598
PRE gap : 0.032806695344304915
CLOSE gap: 0.0009987657925500937
Cube     : 0.05
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_test_contact_detection.py   --scenario center   --sim-backend physx_cpu   --render   --close-steps 60
pre-contact action: 0.75
pre-contact steps : 60
left contact link center : [0.27136218547821045, 0.13105013966560364, 0.1733652502298355]
right contact link center: [0.3257376551628113, 0.13105012476444244, 0.17336544394493103]
computed center          : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
finger vector            : [0.05437546968460083, -1.4901161193847656e-08, 1.9371509552001953e-07]
========================================================================
Day 4 Contact Test
========================================================================
scenario       : center
left center    : [0.27136218547821045, 0.13105013966560364, 0.1733652502298355]
right center   : [0.3257376551628113, 0.13105012476444244, 0.17336544394493103]
gripper center : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
finger distance: 0.05437547
cube target    : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
cube requested : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
cube actual    : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
step=  0 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step=  5 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 10 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 15 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 20 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 25 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 30 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 35 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 40 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 45 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 50 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 55 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
------------------------------------------------------------------------
peak left force   : 0.000000 N
peak right force  : 0.000000 N
left contact steps: 0
right contact steps: 0
both contact steps: 0
report: reports/day4_contact_detection_center.json
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_test_contact_detection.py   --scenario center   --sim-backend physx_cpu   --render   --close-steps 60
pre-contact action: 0.75
pre-contact steps : 60
left contact link center : [0.27136218547821045, 0.13105013966560364, 0.1733652502298355]
right contact link center: [0.3257376551628113, 0.13105012476444244, 0.17336544394493103]
computed center          : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
finger vector            : [0.05437546968460083, -1.4901161193847656e-08, 1.9371509552001953e-07]
========================================================================
Day 4 Contact Test
========================================================================
scenario       : center
left center    : [0.27136218547821045, 0.13105013966560364, 0.1733652502298355]
right center   : [0.3257376551628113, 0.13105012476444244, 0.17336544394493103]
gripper center : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
finger distance: 0.05437547
cube target    : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
cube requested : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
cube actual    : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
step=  0 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step=  5 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 10 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 15 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 20 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 25 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 30 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 35 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 40 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 45 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 50 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 55 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
------------------------------------------------------------------------
peak left force   : 0.000000 N
peak right force  : 0.000000 N
left contact steps: 0
right contact steps: 0
both contact steps: 0
report: reports/day4_contact_detection_center.json
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_test_contact_detection.py   --scenario center   --sim-backend physx_cpu   --render   --close-steps 60
pre-contact action: 0.75
pre-contact steps : 60
left contact link center : [0.27136218547821045, 0.13105013966560364, 0.1733652502298355]
right contact link center: [0.3257376551628113, 0.13105012476444244, 0.17336544394493103]
computed center          : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
finger vector            : [0.05437546968460083, -1.4901161193847656e-08, 1.9371509552001953e-07]
========================================================================
Day 4 Contact Test
========================================================================
scenario       : center
left center    : [0.27136218547821045, 0.13105013966560364, 0.1733652502298355]
right center   : [0.3257376551628113, 0.13105012476444244, 0.17336544394493103]
gripper center : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
finger distance: 0.05437547
cube target    : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
cube requested : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
cube actual    : [0.29854992032051086, 0.13105013966560364, 0.17336535453796387]
step=  0 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step=  5 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 10 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 15 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 20 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 25 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 30 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 35 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 40 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 45 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 50 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
step= 55 L=  0.0000N R=  0.0000N Lcontact=False Rcontact=False grasp=False
------------------------------------------------------------------------
peak left force   : 0.000000 N
peak right force  : 0.000000 N
left contact steps: 0
right contact steps: 0
both contact steps: 0
report: reports/day4_contact_detection_center.json
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_find_precontact_action.py
command      gap[m]      gap[mm]     center xyz
------------------------------------------------------------------------------------------
-1.00      0.185773     185.77    [0.29854961 0.13104973 0.24156055]
-0.95      0.184777     184.78    [0.29854855 0.13104965 0.23841428]
-0.90      0.183517     183.52    [0.29854893 0.13104968 0.23528105]
-0.85      0.181995     181.99    [0.29854921 0.13104967 0.23216584]
-0.80      0.180181     180.18    [0.29854931 0.1310497  0.22907507]
-0.75      0.178019     178.02    [0.29854939 0.1310497  0.22601535]
-0.70      0.175607     175.61    [0.2985493 0.1310497 0.2220462]
-0.65      0.172951     172.95    [0.29854935 0.13104968 0.21808964]
-0.60      0.170055     170.06    [0.29854939 0.13104971 0.2141877 ]
-0.55      0.166926     166.93    [0.29854937 0.13104972 0.21034845]
-0.50      0.163571     163.57    [0.2985493  0.13104975 0.20657993]
-0.45      0.159997     160.00    [0.29854912 0.13104972 0.2028986 ]
-0.40      0.156211     156.21    [0.29854917 0.1310497  0.19930528]
-0.35      0.152221     152.22    [0.29854883 0.13104974 0.19580562]
-0.30      0.148035     148.04    [0.29854875 0.13104974 0.19240812]
-0.25      0.143664     143.66    [0.29854859 0.13104978 0.18911933]
-0.20      0.139115     139.12    [0.29854858 0.13104978 0.18593797]
-0.15      0.134399     134.40    [0.29854846 0.1310498  0.18286563]
-0.10      0.129525     129.53    [0.29854821 0.13104984 0.17994487]
-0.05      0.124504     124.50    [0.29854807 0.13104977 0.17723815]
+0.00      0.119347     119.35    [0.29854806 0.13104977 0.17467237]
+0.05      0.114064     114.06    [0.29854782 0.13104984 0.17225289]
+0.10      0.108636     108.64    [0.29854777 0.13104984 0.16998507]
+0.15      0.103081     103.08    [0.29854772 0.13104985 0.16787368]
+0.20      0.097434      97.43    [0.29854763 0.13104989 0.16592277]
+0.25      0.091708      91.71    [0.29854763 0.13104985 0.164137  ]
+0.30      0.085915      85.92    [0.29854757 0.13104992 0.1625199 ]
+0.35      0.080067      80.07    [0.29854749 0.13104992 0.16107486]
+0.40      0.074177      74.18    [0.29854739 0.13104997 0.15980504]
+0.45      0.068256      68.26    [0.29854743 0.13104993 0.15871327]
+0.50      0.062318      62.32    [0.29854742 0.13104994 0.15780665]
+0.55      0.056376      56.38    [0.29854741 0.13104999 0.15708705]
+0.60      0.050441      50.44    [0.29854748 0.13105    0.15655102]
+0.65      0.044526      44.53    [0.29854754 0.13105005 0.1563707 ]
+0.70      0.038644      38.64    [0.2985475  0.13105009 0.15645244]
+0.75      0.032807      32.81    [0.29854761 0.13105004 0.15671923]
+0.80      0.027027      27.03    [0.29854754 0.13105004 0.15717002]
+0.85      0.021318      21.32    [0.29854766 0.13105012 0.15780402]
+0.90      0.015690      15.69    [0.29854774 0.13105011 0.15862431]
+0.95      0.010157      10.16    [0.2985478  0.13105011 0.15965268]
+1.00      0.000999       1.00    [0.29854773 0.13105006 0.16084645]

========================================================================
Cube size: 0.05 m
Target pre-contact gap: 0.055 m
Recommended command: 0.55
Measured gap: 0.05637572862867374 m
Grasp center: [0.29854741 0.13104999 0.15708705]
========================================================================
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_test_contact_detection.py \
  --scenario center \
  --sim-backend physx_cpu \
  --open-steps 30 \
  --close-steps 80 \
  --render
usage: day4_test_contact_detection.py [-h] [--env-id ENV_ID] [--sim-backend {physx_cpu,physx_cuda}] [--scenario {far,center,left,right}] [--pre-contact-steps PRE_CONTACT_STEPS] [--close-steps CLOSE_STEPS]
                                      [--lateral-offset-factor LATERAL_OFFSET_FACTOR] [--render] [--sleep SLEEP] [--output OUTPUT]
day4_test_contact_detection.py: error: unrecognized arguments: --open-steps 30
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_test_contact_detection.py   --scenario center   --sim-backend physx_cpu   --render   --close-steps 60python scripts/day4_test_contact_detection.py \
  --scenario center \
  --sim-backend physx_cpu \
  --pre-contact-steps 60 \
  --close-steps 10 \
  --render \
  --sleep 0.01 \
  --output reports/day4_contact_detection_center_kinematic.json
usage: day4_test_contact_detection.py [-h] [--env-id ENV_ID] [--sim-backend {physx_cpu,physx_cuda}] [--scenario {far,center,left,right}] [--pre-contact-steps PRE_CONTACT_STEPS] [--close-steps CLOSE_STEPS]
                                      [--lateral-offset-factor LATERAL_OFFSET_FACTOR] [--render] [--sleep SLEEP] [--output OUTPUT]
day4_test_contact_detection.py: error: argument --close-steps: invalid int value: '60python'
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_test_contact_detection.py \
  --scenario center \
  --sim-backend physx_cpu \
  --pre-contact-steps 60 \
  --close-steps 10 \
  --render \
  --sleep 0.01 \
  --output reports/day4_contact_detection_center_kinematic.json
pre-contact action: 0.57
pre-contact steps : 60
Traceback (most recent call last):
  File "/home/ito-marl/ur3e_maniskill/scripts/day4_test_contact_detection.py", line 1001, in <module>
    raise SystemExit(main())
  File "/home/ito-marl/ur3e_maniskill/scripts/day4_test_contact_detection.py", line 571, in main
    target_position = (center.copy())
UnboundLocalError: local variable 'center' referenced before assignment
(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/day4_test_contact_detection.py \
  --scenario center \
  --sim-backend physx_cpu \
  --pre-contact-steps 60 \
  --close-steps 30 \
  --render \
  --sleep 0.01 \
  --output reports/day4_contact_detection_center_kinematic.json
pre-contact action: 0.57
pre-contact steps : 60
left collision center : [0.2618625633177211, 0.1312081895334567, 0.15684962272387454]
right collision center: [0.335237375634902, 0.13089186305694503, 0.15684989886896672]
pre-contact grasp center: [0.29854996947631157, 0.13105002629520085, 0.15684976079642063]
finger distance: 0.07337549417276534
========================================================================
Day 4 Contact Test
========================================================================
scenario       : center
gripper center : [0.29854996947631157, 0.13105002629520085, 0.15684976079642063]
cube target    : [0.29854996947631157, 0.13105002629520085, 0.15684976079642063]
cube requested: [0.29854996947631157, 0.13105002629520085, 0.15684976079642063]
cube actual   : [0.29854997992515564, 0.1310500204563141, 0.15684975683689117]
step=  0 L=139.7975 N R=142.0211 N Lcontact=True Rcontact=True grasp=True
step=  1 L=  0.2917 N R=  0.0000 N Lcontact=True Rcontact=False grasp=False
step=  2 L=  0.2995 N R=  0.0000 N Lcontact=True Rcontact=False grasp=False
step=  3 L= 23.3988 N R= 23.4515 N Lcontact=True Rcontact=True grasp=True
step=  4 L= 28.4291 N R= 28.5027 N Lcontact=True Rcontact=True grasp=True
step=  5 L= 30.8748 N R= 30.9381 N Lcontact=True Rcontact=True grasp=True
step=  6 L= 31.8139 N R= 31.8599 N Lcontact=True Rcontact=True grasp=True
step=  7 L= 31.9968 N R= 32.0291 N Lcontact=True Rcontact=True grasp=True
step=  8 L= 31.8706 N R= 31.8923 N Lcontact=True Rcontact=True grasp=True
step=  9 L= 31.6672 N R= 31.6819 N Lcontact=True Rcontact=True grasp=True
step= 10 L= 31.4843 N R= 31.4955 N Lcontact=True Rcontact=True grasp=True
step= 11 L= 31.3529 N R= 31.3623 N Lcontact=True Rcontact=True grasp=True
step= 12 L= 31.2695 N R= 31.2780 N Lcontact=True Rcontact=True grasp=True
step= 13 L= 31.2241 N R= 31.2328 N Lcontact=True Rcontact=True grasp=True
step= 14 L= 31.2003 N R= 31.2079 N Lcontact=True Rcontact=True grasp=True
step= 15 L= 31.1920 N R= 31.2007 N Lcontact=True Rcontact=True grasp=True
step= 16 L= 31.1903 N R= 31.1994 N Lcontact=True Rcontact=True grasp=True
step= 17 L= 31.1917 N R= 31.2003 N Lcontact=True Rcontact=True grasp=True
step= 18 L= 31.1946 N R= 31.2040 N Lcontact=True Rcontact=True grasp=True
step= 19 L= 31.1969 N R= 31.2050 N Lcontact=True Rcontact=True grasp=True
step= 20 L= 31.1993 N R= 31.2082 N Lcontact=True Rcontact=True grasp=True
step= 21 L= 31.1991 N R= 31.2076 N Lcontact=True Rcontact=True grasp=True
step= 22 L= 31.1995 N R= 31.2081 N Lcontact=True Rcontact=True grasp=True
step= 23 L= 31.1998 N R= 31.2082 N Lcontact=True Rcontact=True grasp=True
step= 24 L= 31.2006 N R= 31.2098 N Lcontact=True Rcontact=True grasp=True
step= 25 L= 31.1994 N R= 31.2083 N Lcontact=True Rcontact=True grasp=True
step= 26 L= 31.2000 N R= 31.2091 N Lcontact=True Rcontact=True grasp=True
step= 27 L= 31.1998 N R= 31.2094 N Lcontact=True Rcontact=True grasp=True
step= 28 L= 31.2000 N R= 31.2089 N Lcontact=True Rcontact=True grasp=True
step= 29 L= 31.1985 N R= 31.2081 N Lcontact=True Rcontact=True grasp=True
------------------------------------------------------------------------
peak left force   : 139.797516 N
peak right force  : 142.021103 N
left contact steps: 30
right contact steps: 28
both contact steps: 28
report: reports/day4_contact_detection_center_kinematic.json

