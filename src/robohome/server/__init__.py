"""家庭服务器：在家里的电脑上跑，手机用浏览器连上来下指令、看进度和带照片的回报、回答机器人的问题。

    robohome serve                         # 只在家里 Wi-Fi 里能用，启动时打印手机要打开的地址（带配对码）
    robohome serve --sim                   # 在 MuJoCo 仿真里做，回报带机器人拍的照片
    robohome serve --planner local         # 用本地模型规划和认标签（默认 Ollama + qwen3:8b）

只用 Python 标准库，不装别的东西也能跑。
"""

from .http import make_server, on_home_network, serve
from .service import Service

__all__ = ["Service", "make_server", "on_home_network", "serve"]
