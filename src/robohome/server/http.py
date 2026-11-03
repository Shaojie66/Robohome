"""网页和接口：手机用浏览器打开，下指令、看进度和回报、回答机器人的问题。

只用 Python 标准库（http.server）。只接受和这台电脑在同一个网段的设备（连着同一个 Wi-Fi 的手机就是）
和本机，别的地址一律拒绝；每个接口都要带配对码（启动时打印的地址里带着，手机打开一次就记住了）。

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
import subprocess
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


# 读不到这台电脑的网卡时，退回去只认常见的家用局域网地址
FALLBACK_NETWORKS = tuple(ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7"))
_INET = re.compile(
    r"inet6?\s+(?:addr:)?([0-9a-fA-F.:]+)(?:%\S+)?(?:/(\d+))?(?:\s+-->\s+\S+)?(?:\s+netmask\s+(\S+)|\s+prefixlen\s+(\d+))?"
)


def _prefix(netmask: str) -> int | None:
    try:
        if netmask.startswith("0x"):
            return int(netmask, 16).bit_count()
        return ipaddress.ip_network(f"0.0.0.0/{netmask}").prefixlen
    except ValueError:
        return None


# VPN 的虚拟网卡（Tailscale、WireGuard 等）：从这些网卡进来的不是家里 Wi-Fi 上的设备，外网访问以后再议
VPN_INTERFACES = re.compile(r"^(utun|tun|tap|wg|tailscale|zt|ppp|ipsec|gif|stf)", re.IGNORECASE)
_IFACE = re.compile(r"^(?:\d+:\s+)?([^\s:]+):?\s")  # "en0: flags=…"（ifconfig）或 "2: wlan0    inet …"（ip -o）


def parse_interfaces(text: str) -> list:
    """从 ifconfig（macOS）或 ip -o addr（Linux）的输出里读出这台电脑的每个地址和它的网段，VPN 网卡不算。"""
    out, name = [], ""
    for line in text.splitlines():
        if line[:1] and not line[0].isspace() and (m := _IFACE.match(line)):
            name = m.group(1)
        if VPN_INTERFACES.match(name):
            continue
        for m in _INET.finditer(line):
            addr, slash, netmask, prefixlen = m.groups()
            prefix = int(slash) if slash else int(prefixlen) if prefixlen else _prefix(netmask) if netmask else None
            if prefix is None:
                continue
            try:
                out.append(ipaddress.ip_interface(f"{addr}/{prefix}"))
            except ValueError:
                continue
    return out


def local_interfaces() -> list | None:
    """这台电脑的网卡地址；读不到（没有 ip 也没有 ifconfig）时返回 None。"""
    for cmd in (["ip", "-o", "addr", "show"], ["ifconfig"], ["/sbin/ifconfig"]):
        try:
            text = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        found = parse_interfaces(text)
        if found:
            return found
    return None


def home_networks(interfaces: list | None) -> tuple:
    """家里的网络 = 这台电脑网卡所在的网段（手机连同一个 Wi-Fi，就在同一个网段里）。

    VPN 那种只有一个地址的点对点网卡（/32）不算：外网访问以后再议。
    """
    if interfaces is None:
        return FALLBACK_NETWORKS
    return tuple(dict.fromkeys(
        i.network for i in interfaces
        if not (i.ip.is_loopback or i.ip.is_link_local) and i.network.prefixlen < i.max_prefixlen
    ))


def on_home_network(address: str, networks=FALLBACK_NETWORKS) -> bool:
    """只认本机、链路本地，以及和这台电脑在同一个网段的地址。"""
    try:
        ip = ipaddress.ip_address(address.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if ip.is_loopback or ip.is_link_local:
        return True
    return any(ip.version == n.version and ip in n for n in networks)


def lan_addresses(interfaces: list | None, networks: tuple) -> list[str]:
    """手机要连的地址：这台电脑在家里网段里的 IPv4 地址，系统默认走的那个排前面。"""
    found = [str(i.ip) for i in interfaces or [] if i.version == 4 and i.network in networks]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.0.1", 9))  # UDP 的 connect 不发包，只是让系统选一个出口网卡
            default = s.getsockname()[0]
        if default in found:
            found.remove(default)
            found.insert(0, default)
        elif interfaces is None and on_home_network(default, networks) and not default.startswith("127."):
            found.insert(0, default)
    except OSError:
        pass
    return list(dict.fromkeys(found))


def page() -> bytes:
    return resources.files(__package__).joinpath("static/index.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version = "robohome"
    service: Service  # 由 make_server 设置
    networks: tuple
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
        if on_home_network(self.client_address[0], self.networks):
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


def make_server(service: Service, host: str = "0.0.0.0", port: int = 8765, networks=None) -> ThreadingHTTPServer:
    """networks 是放行的网段；不给就用这台电脑网卡所在的网段。"""
    if networks is None:
        networks = home_networks(local_interfaces())
    handler = type("BoundHandler", (Handler,), {
        "service": service, "networks": tuple(networks), "failures": {}, "failures_lock": threading.Lock(),
    })
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def _say(text: str) -> None:
    print(text, flush=True)  # 输出重定向到文件时也马上写出去


def serve(service: Service, host: str = "0.0.0.0", port: int = 8765, announce=_say) -> None:
    """开网页服务（后台线程），机器人在主线程里一件件做事（仿真渲染在 Mac 上要在主线程）。Ctrl+C 停止。"""
    interfaces = local_interfaces()
    networks = home_networks(interfaces)
    server = make_server(service, host, port, networks)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    key = service.key
    if host in ("127.0.0.1", "localhost", "::1"):
        urls = [f"http://localhost:{port}/?key={key}"]
    else:
        urls = [f"http://{ip}:{port}/?key={key}" for ip in lan_addresses(interfaces, networks)]
        urls = urls or [f"http://<这台电脑的局域网地址>:{port}/?key={key}"]
        urls.append(f"http://localhost:{port}/?key={key}  （这台电脑自己的浏览器）")
    v4 = [str(n) for n in networks if n.version == 4]
    where = "、".join(v4) if interfaces is not None else "常见的家用局域网地址（读不到这台电脑的网卡）"
    announce(f"机器人管家已经开好（{service.world.home.name}，台账在 {service.store.root}）")
    announce("用连着同一个 Wi-Fi 的手机打开：\n  " + "\n  ".join(urls))
    announce(f"配对码 {key}。只接受和这台电脑在同一个网段的设备（{where or '无'}）和本机。")
    announce("按 Ctrl+C 停止，台账和没做完的任务都会留着。")
    stop = threading.Event()
    try:
        service.run_forever(stop)
    except KeyboardInterrupt:
        announce("\n停下了。下次用同样的 --data-dir 启动，接着用。")
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
