#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path
from typing import Any
import gymnasium as gym
import numpy as np
import yaml
REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT))

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/ur3e_pick_lift_day1.yaml'))
    p.add_argument('--env-id',default=None)
    p.add_argument('--register-module',default='envs.ur3e_reach')
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--render',action='store_true')
    p.add_argument('--output',type=Path,default=Path('reports/day1_ur3e_inspection.json'))
    return p.parse_args()

def to_numpy(v:Any):
    if hasattr(v,'detach'): v=v.detach()
    if hasattr(v,'cpu'): v=v.cpu()
    if hasattr(v,'numpy'): v=v.numpy()
    return np.asarray(v)

def first(v:Any):
    a=to_numpy(v)
    return a[0] if a.ndim>=2 and a.shape[0]==1 else a

def active_joints(robot):
    if hasattr(robot,'active_joints'): return list(robot.active_joints)
    if hasattr(robot,'get_active_joints'): return list(robot.get_active_joints())
    raise AttributeError('active_jointsを取得できません')

def zero_action(space):
    a=space.sample()
    if hasattr(a,'zero_'): a.zero_(); return a
    return np.zeros_like(a)

def position(owner):
    pose=owner.pose
    if hasattr(pose,'p'): return first(pose.p).astype(float)
    if hasattr(pose,'raw_pose'): return first(pose.raw_pose)[:3].astype(float)
    raise AttributeError('pose位置を取得できません')

def main():
    a=parse_args(); cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    importlib.import_module(a.register_module)
    env_id=a.env_id or cfg['project']['env_id']
    env=gym.make(env_id,num_envs=1,obs_mode=cfg['project']['obs_mode'],control_mode=cfg['project']['control_mode'],render_mode='human' if a.render else None)
    try:
        obs,_=env.reset(seed=int(cfg['project']['seed']))
        be=env.unwrapped; agent=be.agent; robot=agent.robot
        joints=active_joints(robot); names=[j.name for j in joints]
        q0=first(robot.get_qpos()).astype(float); qv0=first(robot.get_qvel()).astype(float); qlim=first(robot.get_qlimits()).astype(float)
        tcp0=position(agent.tcp)
        try: base=position(robot)
        except Exception: base=np.asarray(cfg['robot']['base_position'],dtype=float)
        cube=np.asarray(cfg['cube']['position'],dtype=float)
        grasp=cube+np.asarray(cfg['task']['tcp_grasp_offset'],dtype=float)
        pre=grasp+np.asarray([0,0,cfg['task']['pregrasp_clearance']],dtype=float)
        lift=grasp+np.asarray([0,0,cfg['task']['lift_height']],dtype=float)
        for _ in range(a.steps):
            obs,reward,terminated,truncated,info=env.step(zero_action(env.action_space))
            if a.render: env.render()
        q1=first(robot.get_qpos()).astype(float); drift=np.abs(q1-q0)
        ashape=list(env.action_space.shape); oshape=list(getattr(obs,'shape',[])); warnings=[]
        action_low=to_numpy(env.action_space.low).astype(float) if hasattr(env.action_space,'low') else None
        action_high=to_numpy(env.action_space.high).astype(float) if hasattr(env.action_space,'high') else None
        dof=int(cfg['robot']['expected_arm_dof'])
        if len(names)!=dof: warnings.append(f'active joint数={len(names)}、期待={dof}。EZGripper込みなら腕/指関節を分離確認')
        if ashape[-1]!=dof: warnings.append(f'action最終次元={ashape[-1]}、腕6次元と不一致。複合controllerを確認')
        if float(drift.max(initial=0.0))>1e-3: warnings.append('ゼロ差分行動で関節ドリフト>1e-3 rad。PD値・重力補償を確認')
        dist=lambda x,y: float(np.linalg.norm(x-y))
        report={'env_id':env_id,'obs_mode':cfg['project']['obs_mode'],'control_mode':cfg['project']['control_mode'],'observation_shape':oshape,'action_shape':ashape,'action_low':None if action_low is None else action_low.tolist(),'action_high':None if action_high is None else action_high.tolist(),'joint_names':names,'qpos_initial_rad':q0.tolist(),'qvel_initial_rad_s':qv0.tolist(),'qlimits_rad':qlim.tolist(),'robot_base_position_m':base.tolist(),'tcp_initial_position_m':tcp0.tolist(),'zero_action_steps':a.steps,'max_qpos_drift_rad':float(drift.max(initial=0.0)),'targets':{'cube_center_m':cube.tolist(),'grasp_tcp_m':grasp.tolist(),'pregrasp_tcp_m':pre.tolist(),'lift_tcp_m':lift.tolist()},'distances':{'initial_tcp_to_grasp_m':dist(tcp0,grasp),'initial_tcp_to_pregrasp_m':dist(tcp0,pre),'base_to_grasp_m':dist(base,grasp),'base_to_pregrasp_m':dist(base,pre),'base_to_lift_m':dist(base,lift)},'warnings':warnings}
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('='*72); print('Day 1 UR3e実環境検査'); print('='*72)
        print('env_id:',env_id); print('joint names:',names); print('qpos:',q0.tolist()); print('action shape:',ashape); print('action low:',None if action_low is None else action_low.tolist()); print('action high:',None if action_high is None else action_high.tolist()); print('obs shape:',oshape); print('robot base:',base.tolist()); print('initial TCP:',tcp0.tolist()); print('grasp target:',grasp.tolist()); print('pregrasp:',pre.tolist()); print('lift target:',lift.tolist()); print('TCP→grasp:',f"{report['distances']['initial_tcp_to_grasp_m']:.4f} m"); print('最大qpos drift:',f"{report['max_qpos_drift_rad']:.6e} rad")
        for x in warnings: print('[WARN]',x)
        print('report:',a.output)
        return 0
    finally: env.close()

if __name__=='__main__': raise SystemExit(main())
