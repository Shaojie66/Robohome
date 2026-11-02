"""把仿真录下来：任务过程存成 GIF，拍照和现场照存成 PNG。

渲染需要 OpenGL：Mac 上直接能用；没有显示器的 Linux 服务器要装 OSMesa 或 EGL（见 README）。
渲染不可用时只跑物理，不出图。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import mujoco
import numpy as np

log = logging.getLogger(__name__)

HEAD_FOVY = 80  # 头部相机用广角，站在柜子前也能拍到整个柜子

FONTS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]


def _font(size: int):
    from PIL import ImageFont

    for path in FONTS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _camera(lookat, eye) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v = np.asarray(lookat, dtype=float) - np.asarray(eye, dtype=float)
    cam.lookat[:] = lookat
    cam.distance = float(np.linalg.norm(v))
    cam.azimuth = math.degrees(math.atan2(v[1], v[0]))
    cam.elevation = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
    return cam


class Recorder:
    """每隔一段仿真时间截一帧俯视全景，最后存成 GIF；拍照动作另存机器人视角的 PNG。"""

    def __init__(
        self, out_dir: str | Path, fps: int = 10, size: tuple[int, int] = (720, 440), speed: float = 2.0, prefix: str = ""
    ):
        self.out_dir = Path(out_dir)
        self.prefix = prefix  # 照片文件名前缀，同一个目录录好几次时不互相覆盖
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.every = max(1, round(speed / fps / 0.002))  # 每隔多少步截一帧；speed 倍速播放
        self.fps = fps
        self.frames: list = []
        self.renderer: mujoco.Renderer | None = None
        self.failed = False
        self.title = ""
        self.photos = 0

    def _render(self, body, cam) -> np.ndarray | None:
        if self.failed:
            return None
        try:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(body.model, self.size[1], self.size[0])
            self.renderer.update_scene(body.data, camera=cam)
            return self.renderer.render().copy()
        except Exception as e:  # noqa: BLE001 渲染环境千差万别，失败了就只跑物理
            log.warning("渲染不可用，只跑物理：%s", e)
            self.failed = True
            return None

    def overview(self, body) -> mujoco.MjvCamera:
        """斜上方看整个厨房，镜头跟着机器人左右平移一点。"""
        cx, cy = body.scene_center
        lookat = (0.6 * cx + 0.4 * body.base[0], cy + 0.15, 0.75)
        return _camera(lookat, (lookat[0] - 0.25, cy - 3.1, 3.9))

    def tick(self, body) -> None:
        if body.frame_count % self.every:
            return
        img = self._render(body, self.overview(body))
        if img is not None:
            self.frames.append(self._caption(img, body.caption))

    def hold(self, body, seconds: float) -> None:
        """把当前画面停留一会儿。"""
        img = self._render(body, self.overview(body))
        if img is not None:
            self.frames += [self._caption(img, body.caption)] * max(1, round(seconds * self.fps))

    def _caption(self, img: np.ndarray, text: str):
        from PIL import Image, ImageDraw

        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)
        font = _font(22)
        lines = [t for t in (self.title, text) if t]
        y = 10
        for line in lines:
            box = draw.textbbox((14, y), line, font=font)
            draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(255, 255, 255))
            draw.text((14, y), line, fill=(20, 20, 20), font=font)
            y = box[3] + 10
        return im

    def photo(self, body, name: str, look_at=None) -> str | None:
        """机器人头部相机拍一张（拍照技能和出问题时的现场照）。"""
        head = np.array([body.base[0], body.base[1], 1.45]) + 0.1 * body._forward()
        target = look_at if look_at is not None else head + body._forward() + np.array([0, 0, -0.6])
        vis = body.model.vis.global_
        overview_fovy, vis.fovy = vis.fovy, HEAD_FOVY
        try:
            img = self._render(body, _camera(target, head))
        finally:
            vis.fovy = overview_fovy
        if img is None:
            return None
        from PIL import Image

        self.photos += 1
        path = self.out_dir / f"{self.prefix}{self.photos:02d}_{name}.png"
        Image.fromarray(img).save(path)
        return str(path)

    def save_gif(self, path: str | Path, hold_last: float = 1.5) -> str | None:
        if not self.frames:
            return None
        frames = self.frames + [self.frames[-1]] * round(hold_last * self.fps)
        small = [f.convert("P", palette=1, colors=128) for f in frames]  # 1 = ADAPTIVE
        small[0].save(path, save_all=True, append_images=small[1:], duration=round(1000 / self.fps), loop=0, optimize=True)
        return str(path)
