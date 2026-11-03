"""网页和接口：手机用浏览器打开，下指令、看进度和回报、回答机器人的问题。

只用 Python 标准库（http.server）。只接受家里局域网（以及本机）来的连接，外网地址一律拒绝；
每个接口都要带配对码（启动时打印的地址里带着，手机打开一次就记住了）。

    GET  /                          手机网页
    GET  /api/state                 机器人状态、家里有什么、任务列表、等回答的问题
    POST /api/tasks                 {"instruction": "把牛肉拿出来解冻"}
    POST /api/tasks/<id>/cancel     取消排队中或等回答的任务
    POST /api/tasks/<id>/retry      停下了、中断了的任务再做一次
    POST /api/questions/<id>        {"choice": "go", "fields": {"parcel_2": "牛肉"}}
    POST /api/appliances/<id>       {"running": true}  模拟主人自己按家电开关
    GET  /photos/<文件名>?key=…      机器人拍的照片
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import re
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from .service import BusyError, Service

log = logging.getLogger(__name__)

MAX_BODY = 64 * 1024
MAX_FAILURES = 20  # 同一个地址 10 分钟内配对码错这么多次，就先不理它
FAILURE_WINDOW = 600
PHOTO_NAME = re.compile(r"^[\w.-]+\.(png|gif)$")
TYPES = {"png": "image/png", "gif": "image/gif"}


def on_home_network(address: str) -> bool:
    """只认家里局域网和本机：192.168.x.x、10.x.x.x、172.16–31.x.x、本机、链路本地，以及 IPv6 的对应地址。"""
    try:
        ip = ipaddress.ip_address(address.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if ip.is_loopback or ip.is_link_local:
        return True
    if isinstance(ip, ipaddress.IPv4Address):
        return any(ip in ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
    return ip in ipaddress.ip_network("fc00::/7")  # IPv6 的局域网地址


def lan_addresses() -> list[str]:
    """这台电脑在家里局域网里的地址（手机要连的那个）。"""
    found = []
    for probe in ("192.168.0.1", "10.0.0.1", "172.16.0.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 9))  # UDP 的 connect 不发包，只是让系统选一个出口网卡
                ip = s.getsockname()[0]
        except OSError:
            continue
        if ip not in found and on_home_network(ip) and not ip.startswith("127."):
            found.append(ip)
    return found


def page() -> bytes:
    return resources.files(__package__).joinpath("static/index.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version = "robohome"
    service: Service  # 由 make_server 设置
    failures: dict[str, list[float]]
    failures_lock: threading.Lock

    # ---------- 入口 ----------

    def do_GET(self) -> None:
        if not self._from_home():
            return
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, page(), "text/html; charset=utf-8")
        elif url.path == "/api/state":
            if self._authorized(url):
                self._json(HTTPStatus.OK, self.service.state())
        elif url.path.startswith("/photos/"):
            if self._authorized(url):
                self._photo(url.path[len("/photos/"):])
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "没有这个地址"})

    def do_POST(self) -> None:
        if not self._from_home():
            return
        url = urlparse(self.path)
        if not self._authorized(url):
            return
        body = self._body()
        if body is None:
            return
        parts = [p for p in url.path.split("/") if p]
        s = self.service
        try:
            if parts == ["api", "tasks"]:
                self._json(HTTPStatus.CREATED, s.submit(str(body.get("instruction", ""))))
            elif len(parts) == 4 and parts[:2] == ["api", "tasks"] and parts[3] in ("cancel", "retry"):
                result = s.cancel(parts[2]) if parts[3] == "cancel" else s.retry(parts[2])
                self._json(HTTPStatus.OK, result)
            elif len(parts) == 3 and parts[:2] == ["api", "questions"]:
                fields = body.get("fields") or {}
                if not isinstance(fields, dict):
                    raise ValueError("fields 要是对象")
                self._json(HTTPStatus.OK, s.answer(parts[2], str(body.get("choice", "")), fields))
            elif len(parts) == 3 and parts[:2] == ["api", "appliances"]:
                self._json(HTTPStatus.OK, s.set_appliance(parts[2], bool(body.get("running"))))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "没有这个地址"})
        except KeyError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "找不到"})
        except BusyError as e:
            self._json(HTTPStatus.CONFLICT, {"error": str(e)})
        except ValueError as e:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})

    # ---------- 检查 ----------

    def _from_home(self) -> bool:
        if on_home_network(self.client_address[0]):
            return True
        self._json(HTTPStatus.FORBIDDEN, {"error": "只在家里的网络里能用"})
        return False

    def _authorized(self, url) -> bool:
        ip = self.client_address[0]
        now = time.monotonic()
        with self.failures_lock:
            recent = [t for t in self.failures.get(ip, []) if now - t < FAILURE_WINDOW]
            self.failures[ip] = recent
            if len(recent) >= MAX_FAILURES:
                self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "配对码错的次数太多，过一会儿再试"})
                return False
        given = self.headers.get("X-Robohome-Key") or parse_qs(url.query).get("key", [""])[0]
        if given and hmac.compare_digest(given.strip().encode(), self.service.key.encode()):
            return True
        with self.failures_lock:
            self.failures[ip].append(now)
        self._json(HTTPStatus.UNAUTHORIZED, {"error": "配对码不对"})
        return False

    def _body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "请求太大"})
            return None
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "不是 JSON"})
            return None
        if not isinstance(data, dict):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "要是 JSON 对象"})
            return None
        return data

    # ---------- 回复 ----------

    def _photo(self, name: str) -> None:
        folder = self.service.store.photos.resolve()
        path = (folder / name).resolve()
        if not PHOTO_NAME.match(name) or path.parent != folder or not path.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "没有这张照片"})
            return
        self._send(HTTPStatus.OK, path.read_bytes(), TYPES[path.suffix[1:]], cache=True)

    def _json(self, status: HTTPStatus, value) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, data: bytes, kind: str, cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=86400" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        log.debug("%s %s", self.client_address[0], format % args)


def make_server(service: Service, host: str = "0.0.0.0", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {
        "service": service, "failures": {}, "failures_lock": threading.Lock(),
    })
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(service: Service, host: str = "0.0.0.0", port: int = 8765, announce=print) -> None:
    """开网页服务（后台线程），机器人在主线程里一件件做事（仿真渲染在 Mac 上要在主线程）。Ctrl+C 停止。"""
    server = make_server(service, host, port)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    key = service.key
    if host in ("127.0.0.1", "localhost", "::1"):
        urls = [f"http://localhost:{port}/?key={key}"]
    else:
        urls = [f"http://{ip}:{port}/?key={key}" for ip in lan_addresses()] or [f"http://<这台电脑的局域网地址>:{port}/?key={key}"]
        urls.append(f"http://localhost:{port}/?key={key}  （这台电脑自己的浏览器）")
    announce(f"机器人管家已经开好（{service.world.home.name}，台账在 {service.store.root}）")
    announce("用连着同一个 Wi-Fi 的手机打开：\n  " + "\n  ".join(urls))
    announce(f"配对码 {key}，只在家里的网络里能用。按 Ctrl+C 停止，台账和没做完的任务都会留着。")
    stop = threading.Event()
    try:
        service.run_forever(stop)
    except KeyboardInterrupt:
        announce("\n停下了。下次用同样的 --data-dir 启动，接着用。")
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
