"""用宇树 G1 的公开模型算手够得着的范围，结果填进 src/robohome/sim/layout.py 的 G1_REACH。

模型来自 MuJoCo Menagerie（BSD-3-Clause，由宇树公开的 g1_29dof_rev_1_0 整理）：

    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
    cd mujoco_menagerie && git sparse-checkout set unitree_g1 && cd ..
    pip install mujoco scipy
    python scripts/g1_reach.py mujoco_menagerie/unitree_g1/g1.xml

怎么算：
- 两脚踩平不动：左右腿一样，髋、膝、踝前后转，踝的角度让脚底保持水平
- 蹲、弯腰（骨盆前倾不超过 0.5 弧度，腰部三个关节在限位内）、转腰、伸右手（7 个关节在限位内）
- 站得住：手里拿 1 公斤东西时，整体重心落在两只脚围成的范围里，四边各留 2 厘米
- 手的位置取掌心（腕部最后一节往前 8 厘米）；只看身体正前方左右 60 度以内
- 每个高度用 SLSQP 从多个起点求"手离两脚中点水平最远多远"

不算全身动态平衡、手腕朝向和身体与家具的碰撞，所以写进代码时每个高度再减 5 厘米。
"""

from __future__ import annotations

import math
import sys

import mujoco
import numpy as np
from scipy.optimize import minimize

PALM = np.array([0.08, 0.0, 0.0])  # 掌心在腕部最后一节坐标系里的位置
LOAD_KG = 1.0
COM_MARGIN = 0.02
SOLE = 0.035  # 脚上的 site 比脚底高这么多
FOOT_X = (-0.05, 0.12)  # 脚底接触点前后范围（相对踝）
FOOT_Y = 0.03
CONE_DEG = 60
MAX_PELVIS_PITCH = 0.5

ARM = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEGS = [(f"{s}_hip_pitch_joint", f"{s}_knee_joint", f"{s}_ankle_pitch_joint") for s in ("left", "right")]


class G1:
    def __init__(self, path: str):
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.adr = {n: m.joint(n).qposadr[0] for n in ARM + WAIST + [j for leg in LEGS for j in leg]}
        self.lim = {n: tuple(m.jnt_range[m.joint(n).id]) for n in self.adr}
        self.lf, self.rf = m.site("left_foot").id, m.site("right_foot").id
        self.hand = m.body("right_wrist_yaw_link").id
        self.mass = m.body_subtreemass[1]
        # 变量：骨盆前倾、髋、膝、腰三个、手臂七个；踝由"脚底水平"算出来
        self.bounds = [(-0.1, MAX_PELVIS_PITCH), (-2.0, 0.3), (0.0, 2.6)] + [self.lim[n] for n in WAIST + ARM]
        self.ankle = self.lim["left_ankle_pitch_joint"]

    def fk(self, q):
        """返回 (掌心相对两脚中点地面的位置, 重心离支撑边界的四个余量, 踝角)。"""
        m, d = self.m, self.d
        p, h, k = q[:3]
        a = -(p + h + k)
        d.qpos[:] = 0
        d.qpos[0:3] = (0, 0, 0.793)
        d.qpos[3:7] = (math.cos(p / 2), 0, math.sin(p / 2), 0)
        for hp, kn, an in LEGS:
            d.qpos[self.adr[hp]], d.qpos[self.adr[kn]], d.qpos[self.adr[an]] = h, k, a
        for n, v in zip(WAIST + ARM, q[3:]):
            d.qpos[self.adr[n]] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        left, right = d.site_xpos[self.lf].copy(), d.site_xpos[self.rf].copy()
        floor = min(left[2], right[2]) - SOLE
        mid = (left + right) / 2
        palm = d.xpos[self.hand] + d.xmat[self.hand].reshape(3, 3) @ PALM
        com = (self.mass * d.subtree_com[1] + LOAD_KG * palm) / (self.mass + LOAD_KG)
        rel = palm - np.array([mid[0], mid[1], floor])
        margins = np.array([
            com[0] - (mid[0] + FOOT_X[0] + COM_MARGIN),
            (mid[0] + FOOT_X[1] - COM_MARGIN) - com[0],
            com[1] - (right[1] - FOOT_Y + COM_MARGIN),
            (left[1] + FOOT_Y - COM_MARGIN) - com[1],
        ])
        return rel, margins, a

    def random_pose(self, rng):
        return np.array([rng.uniform(lo, hi) for lo, hi in self.bounds])

    def valid(self, q, zt=None) -> bool:
        rel, margins, a = self.fk(q)
        ok = (margins >= -1e-4).all() and self.ankle[0] - 1e-4 <= a <= self.ankle[1] + 1e-4 and rel[2] > 0.02
        ok = ok and math.tan(math.radians(CONE_DEG)) * rel[0] >= abs(rel[1]) - 1e-4
        return ok and (zt is None or abs(rel[2] - zt) < 1e-3)


def main(path: str) -> None:
    g1 = G1(path)
    rng = np.random.default_rng(0)
    seeds = []
    for _ in range(300_000):
        q = g1.random_pose(rng)
        if g1.valid(q):
            seeds.append((q, g1.fk(q)[0]))
    print(f"随机姿态里站得住的：{len(seeds)}", file=sys.stderr)
    Q = np.array([q for q, _ in seeds])
    P = np.array([p for _, p in seeds])
    tan = math.tan(math.radians(CONE_DEG))
    print("高度(米)  水平最远(米)")
    for zt in np.arange(0.05, 1.66, 0.05):
        cons = [
            {"type": "eq", "fun": lambda q, zt=zt: g1.fk(q)[0][2] - zt},
            {"type": "ineq", "fun": lambda q: g1.fk(q)[1]},
            {"type": "ineq", "fun": lambda q: np.array([g1.fk(q)[2] - g1.ankle[0], g1.ankle[1] - g1.fk(q)[2]])},
            {"type": "ineq", "fun": lambda q: tan * g1.fk(q)[0][0] - abs(g1.fk(q)[0][1])},
        ]
        order = np.argsort(np.abs(P[:, 2] - zt) * 3 - np.hypot(P[:, 0], P[:, 1]))[:15]
        best = 0.0
        for i in order:
            r = minimize(lambda q: -math.hypot(*g1.fk(q)[0][:2]), Q[i], method="SLSQP", bounds=g1.bounds,
                         constraints=cons, options={"maxiter": 300, "ftol": 1e-7})
            if g1.valid(r.x, zt):
                best = max(best, math.hypot(*g1.fk(r.x)[0][:2]))
        print(f"{zt:.2f}  {best:.3f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
