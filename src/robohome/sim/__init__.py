"""物理仿真（MuJoCo）：把技能库的"模拟实现"换成在物理世界里真的做一遍，规划和调度代码不动。

    pip install 'robohome[sim]'
    robohome sim-check                      # 检查规范文件里的三维数据：放得进、够得着
    robohome mvp --sim                      # 五个任务在仿真里跑
    robohome mvp --sim --faults             # 故障由物理去体现：门真的拉不开、东西真的掉下去
    robohome mvp --sim --record out/        # 录成 GIF

几何检查（layout.py）只用规范数据，不需要装 MuJoCo。
"""

from __future__ import annotations

import os
import sys

# 没有显示器的 Linux（服务器、容器）默认用软件渲染；必须在 import mujoco 之前设置
if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

INSTALL_HINT = "仿真需要先安装 MuJoCo：pip install 'robohome[sim]'"


def available() -> bool:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return True
