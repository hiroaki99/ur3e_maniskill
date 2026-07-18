(maniskill) ito-marl@ubuntu:~/ur3e_maniskill$ python scripts/run_ppo_ur3e.py   --env_id="UR3eReach-v0"   --num_envs=8   --update_epochs=4   --num_minibatches=4   --total_timesteps=100000   --eval_freq=10   --num-steps=20
Saving eval videos to runs/UR3eReach-v0__ppo__1__1784380870/videos
Running training
####
args.num_iterations=625 args.num_envs=8 args.num_eval_envs=8
args.minibatch_size=40 args.batch_size=160 args.update_epochs=4
####
Epoch: 1, global_step=0
Evaluating
Evaluated 400 steps resulting in 0 episodes
model saved to runs/UR3eReach-v0__ppo__1__1784380870/ckpt_1.pt
SPS: 63
Epoch: 2, global_step=160
SPS: 112
Epoch: 3, global_step=320
SPS: 151
Epoch: 4, global_step=480
SPS: 183
Epoch: 5, global_step=640
SPS: 210
Epoch: 6, global_step=800
SPS: 233
Epoch: 7, global_step=960
SPS: 252
Epoch: 8, global_step=1120
SPS: 269
Epoch: 9, global_step=1280
SPS: 284
Epoch: 10, global_step=1440
SPS: 296
Epoch: 11, global_step=1600
Evaluating
Evaluated 400 steps resulting in 0 episodes
model saved to runs/UR3eReach-v0__ppo__1__1784380870/ckpt_11.pt
