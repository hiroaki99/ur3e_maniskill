#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any
import yaml

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/ur3e_pick_lift.yaml'))
    p.add_argument('--output-dir',type=Path,default=Path('reports'))
    return p.parse_args()

def require(m:dict[str,Any],k:str,path:str):
    if k not in m: raise ValueError(f'必須項目がありません: {path}.{k}')
    return m[k]

def vec(v:Any,n:int,path:str):
    if not isinstance(v,list) or len(v)!=n: raise ValueError(f'{path} は長さ{n}のlistである必要があります')
    return [float(x) for x in v]

def add(a,b): return [x+y for x,y in zip(a,b)]
def norm(v): return math.sqrt(sum(x*x for x in v))

def main():
    a=parse_args()
    if not a.config.exists(): raise FileNotFoundError(a.config)
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    errors=[]; warnings=[]
    try:
        project=require(cfg,'project','root'); robot=require(cfg,'robot','root'); table=require(cfg,'table','root')
        cube=require(cfg,'cube','root'); task=require(cfg,'task','root'); residual=require(cfg,'residual','root'); safety=require(cfg,'safety','root')
        expected_arm_dof=int(require(robot,'expected_arm_dof','robot'))
        table_z=float(require(table,'surface_z','table'))
        cube_size=float(require(cube,'size','cube')); cube_mass=float(require(cube,'mass','cube'))
        cube_position=vec(require(cube,'position','cube'),3,'cube.position')
        cube_q=vec(require(cube,'orientation_wxyz','cube'),4,'cube.orientation_wxyz')
        base_position=vec(require(robot,'base_position','robot'),3,'robot.base_position')
        lift_height=float(require(task,'lift_height','task')); hold_steps=int(require(task,'hold_steps','task'))
        max_episode_steps=int(require(task,'max_episode_steps','task'))
        grasp_offset=vec(require(task,'tcp_grasp_offset','task'),3,'task.tcp_grasp_offset')
        pregrasp_clearance=float(require(task,'pregrasp_clearance','task'))
        tolerance=float(require(task,'success_distance_tolerance','task'))
        alpha=float(require(residual,'alpha','residual'))
        max_joint_delta=float(require(residual,'max_joint_delta_rad','residual'))
        max_base_distance=float(require(safety,'max_base_to_target_distance','safety'))
        min_target_z=float(require(safety,'min_target_z','safety'))
        if expected_arm_dof!=6: errors.append(f'expected_arm_dof={expected_arm_dof}; UR3e腕は6自由度想定')
        if not 0.01<=cube_size<=0.10: warnings.append(f'cube.size={cube_size:.3f} m は要確認')
        if not 0.01<=cube_mass<=1.0: warnings.append(f'cube.mass={cube_mass:.3f} kg は要確認')
        expected_cube_z=table_z+cube_size/2
        if abs(cube_position[2]-expected_cube_z)>1e-6: errors.append(f'cube z不一致: 設定={cube_position[2]:.6f},期待={expected_cube_z:.6f}')
        if abs(norm(cube_q)-1.0)>1e-6: errors.append('Quaternionが正規化されていません')
        if lift_height<=0: errors.append('lift_heightは正である必要があります')
        if hold_steps<=0: errors.append('hold_stepsは1以上')
        if max_episode_steps<=hold_steps: errors.append('max_episode_stepsはhold_stepsより大きくする')
        if not 0<alpha<=1: errors.append('alphaは0より大きく1以下')
        if not 0<max_joint_delta<=0.10: warnings.append(f'max_joint_delta={max_joint_delta:.3f} rad/stepは要確認')
        if tolerance<=0: errors.append('success_distance_toleranceは正')
        grasp_tcp=add(cube_position,grasp_offset)
        pregrasp_tcp=add(grasp_tcp,[0,0,pregrasp_clearance])
        lift_tcp=add(grasp_tcp,[0,0,lift_height])
        for name,target in [('grasp_tcp',grasp_tcp),('pregrasp_tcp',pregrasp_tcp),('lift_tcp',lift_tcp)]:
            d=norm([target[i]-base_position[i] for i in range(3)])
            if d>max_base_distance: warnings.append(f'{name} base距離 {d:.3f}m > 簡易上限 {max_base_distance:.3f}m。IK確認が必要')
            if target[2]<min_target_z: errors.append(f'{name}.z={target[2]:.3f}mが安全下限未満')
        resolved={'source_config':str(a.config),'environment':{'env_id':str(require(project,'env_id','project')),'obs_mode':str(require(project,'obs_mode','project')),'control_mode':str(require(project,'control_mode','project')),'seed':int(require(project,'seed','project'))},'cube':{'size_m':cube_size,'mass_kg':cube_mass,'position_m':cube_position,'orientation_wxyz':cube_q},'targets':{'grasp_tcp_m':grasp_tcp,'pregrasp_tcp_m':pregrasp_tcp,'lift_tcp_m':lift_tcp},'success':{'lift_height_m':lift_height,'hold_steps':hold_steps,'distance_tolerance_m':tolerance,'max_episode_steps':max_episode_steps},'residual':{'alpha':alpha,'max_joint_delta_rad':max_joint_delta,'effective_max_residual_rad':alpha*max_joint_delta},'errors':errors,'warnings':warnings}
    except Exception as e:
        errors.append(str(e)); resolved={'source_config':str(a.config),'errors':errors,'warnings':warnings}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    jp=a.output_dir/'day1_pick_lift_spec.json'; mp=a.output_dir/'day1_pick_lift_spec.md'
    jp.write_text(json.dumps(resolved,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Day 1 Pick-and-Lift仕様検証結果','',f'- 設定ファイル: `{a.config}`',f'- エラー数: {len(errors)}',f'- 警告数: {len(warnings)}','']
    if 'targets' in resolved:
        lines += ['## 導出したTCP目標点','',f"- grasp: `{resolved['targets']['grasp_tcp_m']}` m",f"- pre-grasp: `{resolved['targets']['pregrasp_tcp_m']}` m",f"- lift: `{resolved['targets']['lift_tcp_m']}` m",'']
    if errors: lines += ['## エラー','']+[f'- {x}' for x in errors]+['']
    if warnings: lines += ['## 警告','']+[f'- {x}' for x in warnings]+['']
    if not errors: lines += ['## 判定','','数値仕様の整合性チェックに合格。実到達性は環境起動スクリプトで確認する。','']
    mp.write_text('\n'.join(lines),encoding='utf-8')
    print('='*72); print('Day 1 Pick-and-Lift仕様検証'); print('='*72)
    if 'targets' in resolved:
        print('grasp TCP   :',resolved['targets']['grasp_tcp_m']); print('pregrasp TCP:',resolved['targets']['pregrasp_tcp_m']); print('lift TCP    :',resolved['targets']['lift_tcp_m']); print('実効残差上限:',resolved['residual']['effective_max_residual_rad'],'rad/step')
    for x in warnings: print('[WARN]',x)
    for x in errors: print('[ERROR]',x)
    print('JSON:',jp); print('Markdown:',mp)
    return 1 if errors else 0

if __name__=='__main__': raise SystemExit(main())