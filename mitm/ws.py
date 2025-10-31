# im_ws_decompress.py
# mitmproxy addon: 对指定目标域名的 websocket 消息尝试 permessage-deflate 解压并写入 jsonl 日志
# 特性：
#  - 支持环境变量覆盖 OUTFILE / FALLBACK_OUTFILE / TARGET_HOST
#  - 展开 ~，创建目录，使用 os.open(O_APPEND) 做原子追加写入
#  - 启动时进行 test 写入并记录 mitmproxy 日志
#  - 更丰富的 ctx.log 调试信息
#  - 改进的注入/send 实现，支持多种 mitmproxy API、raw socket sendall、client->server 掩码

from mitmproxy import ctx, http
from mitmproxy.websocket import WebSocketMessage
import json, base64, time, zlib, re, os, getpass, traceback, errno, struct

# 额外导入用于兼容不同签名的 WebSocketMessage
from mitmproxy.websocket import WebSocketMessage as MitmWebSocketMessage

# ---------- 配置（可由环境变量覆盖） ----------
DEFAULT_OUTFILE = "/Users/v_shemingdong/im_synclubaichat_ws.jsonl"  # 你原先的首选路径
DEFAULT_FALLBACK = "/tmp/im_synclubaichat_ws.jsonl"
DEFAULT_TARGET_HOST = "im.synclubaichat.com"

OUTFILE = os.environ.get("IM_WS_OUTFILE", DEFAULT_OUTFILE)
FALLBACK_OUTFILE = os.environ.get("IM_WS_FALLBACK_OUTFILE", DEFAULT_FALLBACK)
TARGET_HOST = os.environ.get("IM_WS_TARGET_HOST", DEFAULT_TARGET_HOST)

# ---------- 帮助函数：路径解析与写测试 ----------
def _resolve_outfile(path_candidate):
    if not path_candidate:
        return None
    path = os.path.expanduser(path_candidate)
    path = os.path.abspath(path)
    outdir = os.path.dirname(path)
    try:
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            ctx.log.info(f"[WS-DECOMP] ensured directory exists: {outdir}")
    except Exception as e:
        ctx.log.warn(f"[WS-DECOMP] create dir {outdir} failed: {e}")
    return path

def _log_env_and_test_write():
    """
    启动时按优先级尝试写入（环境变量覆盖 -> 默认路径 -> fallback）
    使用低级 os.open + os.O_APPEND 以减小并发问题。
    返回第一个可写路径或 None。
    """
    try:
        user = getpass.getuser()
    except Exception:
        user = "<unknown>"
    try:
        cwd = os.getcwd()
    except Exception:
        cwd = "<unknown>"

    candidates = []
    if os.environ.get("IM_WS_OUTFILE"):
        candidates.append(os.environ.get("IM_WS_OUTFILE"))
    candidates.append(DEFAULT_OUTFILE)
    candidates.append(FALLBACK_OUTFILE)

    ctx.log.info(f"[WS-DECOMP] startup env -> user: {user} cwd: {cwd} candidates: {candidates}")

    for cand in candidates:
        path = _resolve_outfile(cand)
        if not path:
            continue
        try:
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            # mode 0o644
            fd = os.open(path, flags, 0o644)
            try:
                with os.fdopen(fd, "a", encoding="utf-8") as f:
                    f.write(f"__test__ {time.time()}\n")
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        # fsync 失败不是致命，但记录
                        ctx.log.debug(f"[WS-DECOMP] fsync failed on test write for {path}")
                ctx.log.info(f"[WS-DECOMP] test write succeeded to {path}")
                return path
            except Exception as e:
                # 若 fdopen 写入失败，确保 fd 被关闭
                try:
                    os.close(fd)
                except Exception:
                    pass
                ctx.log.error(f"[WS-DECOMP] fd write failed for {path}: {e}")
        except OSError as oe:
            ctx.log.error(f"[WS-DECOMP] open {path} failed: {oe} (errno {getattr(oe,'errno',None)})")
        except Exception as e:
            ctx.log.error(f"[WS-DECOMP] unexpected error opening {path}: {e}")
    ctx.log.error("[WS-DECOMP] no writable outfile available after trying candidates")
    return None

# 在模块加载时执行检测并确定最终 OUTFILE
_actual_outfile = _log_env_and_test_write()
if _actual_outfile:
    OUTFILE = _actual_outfile
else:
    ctx.log.error("[WS-DECOMP] no writable outfile available. append_record will still attempt original OUTFILE.")

# ---------- 写入函数：使用低级 open + append，带 fallback ----------
def append_record(rec):
    """
    将记录追加到 OUTFILE（或回退路径）。使用低级 open + fsync，捕获并记录详细异常。
    """
    global OUTFILE
    try:
        if not OUTFILE:
            raise RuntimeError("OUTFILE is not set or not writable")
        # 再次确保目录存在（race 再次检查）
        od = os.path.dirname(OUTFILE)
        if od and not os.path.exists(od):
            try:
                os.makedirs(od, exist_ok=True)
            except Exception as e:
                ctx.log.warn(f"[WS-DECOMP] unable to create dir {od}: {e}")

        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        fd = os.open(OUTFILE, flags, 0o644)
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    ctx.log.debug(f"[WS-DECOMP] fsync failed for {OUTFILE}")
            ctx.log.debug(f"[WS-DECOMP] append_record wrote to {OUTFILE}")
            return
        except Exception:
            # 若写入失败，确保 fd 关闭
            try:
                os.close(fd)
            except Exception:
                pass
            raise
    except Exception as e:
        ctx.log.error(f"[WS-DECOMP] append_record error writing to {OUTFILE}: {e}")
        ctx.log.error(traceback.format_exc())
        # 尝试写入 fallback（若尚未使用）
        if OUTFILE != FALLBACK_OUTFILE:
            try:
                flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
                fd2 = os.open(FALLBACK_OUTFILE, flags, 0o644)
                with os.fdopen(fd2, "a", encoding="utf-8") as f2:
                    f2.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f2.flush()
                    try:
                        os.fsync(f2.fileno())
                    except Exception:
                        pass
                ctx.log.info(f"[WS-DECOMP] append_record fallback wrote to {FALLBACK_OUTFILE}")
                OUTFILE = FALLBACK_OUTFILE
            except Exception as e2:
                ctx.log.error(f"[WS-DECOMP] append_record fallback also failed: {e2}")
                ctx.log.error(traceback.format_exc())

# ---------- permessage-deflate flags 解析 ----------
def parse_extensions(header_value: str):
    res = {"permessage-deflate": False, "server_no_context_takeover": False, "client_no_context_takeover": False}
    if not header_value:
        return res
    try:
        hv = header_value.lower()
        if "permessage-deflate" in hv:
            res["permessage-deflate"] = True
            if "server_no_context_takeover" in hv:
                res["server_no_context_takeover"] = True
            if "client_no_context_takeover" in hv:
                res["client_no_context_takeover"] = True
    except Exception as e:
        ctx.log.debug(f"[WS-DECOMP] parse_extensions error: {e}")
    return res

# ---------- websocket handshake: 初始化 decompressor（如果需要） ----------
def websocket_handshake(flow):
    try:
        resp = getattr(flow, "handshake_response", None)
        req = getattr(flow, "handshake_request", None)
        host = ""
        if req and getattr(req, "host", None):
            host = req.host
        elif resp and getattr(resp, "headers", None):
            host = resp.headers.get("host", "")
        else:
            host = ""

        # 只对目标域名生效
        if TARGET_HOST not in (host or ""):
            return

        ext = None
        if resp and getattr(resp, "headers", None):
            # mitmproxy 的 headers 可能是字典样式
            ext = resp.headers.get("sec-websocket-extensions", "")
        if not ext and req and getattr(req, "headers", None):
            ext = req.headers.get("sec-websocket-extensions", "")

        flags = parse_extensions(ext)
        # ensure metadata dict exists
        if getattr(flow, "metadata", None) is None:
            try:
                flow.metadata = {}
            except Exception:
                pass
        # 当没有 context takeover 时，不保留 decompressor
        flow.metadata["pmd_flags"] = flags
        flow.metadata["pmd_decomp_srv"] = zlib.decompressobj(-15) if not flags.get("server_no_context_takeover") else None
        flow.metadata["pmd_decomp_cli"] = zlib.decompressobj(-15) if not flags.get("client_no_context_takeover") else None
        ctx.log.info(f"[WS-DECOMP] Handshake host={host} flow={getattr(flow,'id',None)} permessage-deflate flags: {flags}")
    except Exception as e:
        ctx.log.error(f"[WS-DECOMP] websocket_handshake error: {e}")
        ctx.log.error(traceback.format_exc())

# ---------- 尝试解压 permessage-deflate（支持有/无 context takeover） ----------
def try_decompress_pmd(flow, data_bytes: bytes, from_client: bool):
    try:
        flags = {}
        if getattr(flow, "metadata", None):
            flags = flow.metadata.get("pmd_flags", {}) if getattr(flow, "metadata", None) else {}
        no_ctx = flags.get("server_no_context_takeover") if not from_client else flags.get("client_no_context_takeover")
        trailer = b'\x00\x00\xff\xff'
        if no_ctx:
            # 没有 context takeover：每个消息独立解压
            try:
                out = zlib.decompress(data_bytes + trailer, -15)
                return True, out.decode("utf-8", errors="replace"), False
            except Exception:
                try:
                    out = zlib.decompress(data_bytes)
                    return True, out.decode("utf-8", errors="replace"), False
                except Exception:
                    return False, None, True
        else:
            # 有 context takeover：尝试使用保留的 decompressor
            key = "pmd_decomp_srv" if not from_client else "pmd_decomp_cli"
            dobj = None
            try:
                dobj = flow.metadata.get(key) if getattr(flow, "metadata", None) else None
            except Exception:
                dobj = None
            if dobj is None:
                try:
                    dobj = zlib.decompressobj(-15)
                    if getattr(flow, "metadata", None) is not None:
                        flow.metadata[key] = dobj
                except Exception:
                    dobj = None
            try:
                if dobj is not None:
                    out = dobj.decompress(data_bytes + trailer)
                else:
                    out = zlib.decompress(data_bytes + trailer, -15)
                return True, out.decode("utf-8", errors="replace"), False
            except Exception:
                try:
                    out = zlib.decompress(data_bytes + trailer, -15)
                    return True, out.decode("utf-8", errors="replace"), False
                except Exception:
                    return False, None, True
    except Exception as e:
        ctx.log.error(f"[WS-DECOMP] try_decompress_pmd error: {e}")
        ctx.log.error(traceback.format_exc())
        return False, None, True

# ---------- 通用 wrapper：兼容不同 mitmproxy 调用方式 ----------
from mitmproxy.http import HTTPFlow
from mitmproxy.websocket import WebSocketMessage as MitmWebSocketMessage

def websocket_message(*args, **kwargs):
    """
    更鲁棒的 wrapper：兼容多种 mitmproxy 版本/调用方式，包括单参数为 HTTPFlow 的情况。
    当接收到 HTTPFlow 时，会尝试从 flow.websocket.messages 中提取 websocket 消息并逐条处理。
    """
    try:
        flow = None
        message = None

        # 优先从 kwargs 读取
        if "flow" in kwargs and "message" in kwargs:
            flow = kwargs.get("flow")
            message = kwargs.get("message")
        else:
            # 根据位置参数数量做兼容处理
            if len(args) >= 3:
                # 常见绑定方法 self, flow, message
                _, flow, message = args[:3]
            elif len(args) == 2:
                flow, message = args
            elif len(args) == 1:
                single = args[0]
                # 1) 如果是 tuple/list 并长度为2，解包
                if isinstance(single, (tuple, list)) and len(single) == 2:
                    flow, message = single
                else:
                    # 2) 如果是 HTTPFlow（mitmproxy 把整个 flow 传进来），尝试提取 websocket 消息
                    if isinstance(single, HTTPFlow):
                        flow = single
                        # 如果 flow 有 websocket 属性并包含 messages，则处理其 messages
                        ws = getattr(flow, "websocket", None)
                        if ws is not None:
                            msgs = getattr(ws, "messages", None)
                            if msgs:
                                ctx.log.info(f"[WS-DECOMP] websocket_message received HTTPFlow with {len(msgs)} websocket.messages; processing them.")
                                # 逐条处理（注意：可能会重复处理已处理的消息，若需避免请用 flow.metadata 做标记）
                                for m in msgs:
                                    try:
                                        _websocket_message_impl(flow, m)
                                    except Exception as e:
                                        ctx.log.error(f"[WS-DECOMP] error processing websocket message from HTTPFlow: {e}")
                                        ctx.log.error(traceback.format_exc())
                                return
                            else:
                                # 旧版本/不同结构，尝试直接当 message 处理（如果对象看起来像 message）
                                if hasattr(single, "message") and isinstance(getattr(single, "message"), MitmWebSocketMessage):
                                    message = getattr(single, "message")
                                else:
                                    ctx.log.warn(f"[WS-DECOMP] HTTPFlow has websocket attr but no messages list. keys: {dir(ws)}")
                                    return
                        else:
                            # 没有 websocket 属性，尝试把 single 当作 message-like 对象（包含 flow）
                            if hasattr(single, "message") and hasattr(single, "flow"):
                                flow = getattr(single, "flow")
                                message = getattr(single, "message")
                            else:
                                ctx.log.warn(f"[WS-DECOMP] single arg is HTTPFlow but no websocket attr; returning. flow id={getattr(single,'id',None)}")
                                return
                    else:
                        # 3) 如果对象有 flow 属性并且对象本身像 message（有 content/is_text），则把它当作 message
                        if hasattr(single, "flow") and (hasattr(single, "content") or hasattr(single, "is_text") or hasattr(single, "from_client")):
                            flow = getattr(single, "flow")
                            message = single
                        elif hasattr(single, "message") and hasattr(single, "flow"):
                            flow = getattr(single, "flow")
                            message = getattr(single, "message")
                        else:
                            flow = kwargs.get("flow", None)
                            message = kwargs.get("message", None)

        # 最终检查
        if not flow or not message:
            ctx.log.warn(f"[WS-DECOMP] websocket_message could not normalize args. len(args)={len(args)} kwargs_keys={list(kwargs.keys())} "
                         f"arg0_type={type(args[0]) if len(args)>0 else None}")
            return

        # 如果 message 不是 mitmproxy.websocket.WebSocketMessage 的实例，但看起来像（有 content/is_text），可以仍然传入处理函数
        _websocket_message_impl(flow, message)

    except Exception as e:
        ctx.log.error(f"[WS-DECOMP] websocket_message wrapper error: {e}")
        ctx.log.error(traceback.format_exc())



# ---------- 真实处理函数 ----------
def _websocket_message_impl(flow, message: WebSocketMessage):
    try:
        # Only target domain
        host = ""
        try:
            if getattr(flow, "handshake_request", None) and getattr(flow.handshake_request, "host", None):
                host = flow.handshake_request.host
            elif getattr(flow, "request", None) and getattr(flow.request, "host", None):
                host = flow.request.host
        except Exception:
            host = ""
        # 仅处理匹配 TARGET_HOST 的 flow
        if TARGET_HOST not in (host or ""):
            return

        # message may be mitmproxy.websocket.WebSocketMessage
        is_text = bool(getattr(message, "is_text", False))
        from_client = bool(getattr(message, "from_client", False))
        direction = "client->server" if from_client else "server->client"
        # content can be bytes or str in different mitmproxy versions
        raw_content = getattr(message, "content", b"")
        if isinstance(raw_content, str):
            msg_bytes = raw_content.encode("utf-8")
        elif isinstance(raw_content, (bytes, bytearray)):
            msg_bytes = raw_content
        else:
            msg_bytes = b""

        rec = {
            "ts": time.time(),
            "flow_id": getattr(flow, "id", None),
            "host": host,
            "direction": direction,
            "is_text": is_text,
            "message_text": None,
            "message_base64": None,
            "decompressed": False,
            "decompress_error": None
        }

        # 在进入处理前记录下基本信息（便于诊断）
        ctx.log.debug(f"[WS-DECOMP] recv WS {direction} host={host} flow={getattr(flow,'id',None)} is_text={is_text} len={len(msg_bytes)}")

        if is_text:
            try:
                if isinstance(raw_content, (bytes, bytearray)):
                    rec["message_text"] = raw_content.decode("utf-8")
                else:
                    rec["message_text"] = str(raw_content)
                rec["decompressed"] = False
            except Exception:
                ok, txt, used_b64 = try_decompress_pmd(flow, msg_bytes, from_client)
                if ok:
                    rec["message_text"] = txt
                    rec["decompressed"] = True
                else:
                    rec["message_base64"] = base64.b64encode(msg_bytes).decode()
                    rec["decompress_error"] = "text decode failed and pmd decompress failed"
        else:
            ok, txt, used_b64 = try_decompress_pmd(flow, msg_bytes, from_client)
            if ok:
                rec["message_text"] = txt
                rec["decompressed"] = True
            else:
                try:
                    rec["message_text"] = msg_bytes.decode("utf-8")
                    rec["decompressed"] = False
                except Exception:
                    rec["message_base64"] = base64.b64encode(msg_bytes).decode()
                    rec["decompress_error"] = "decompress failed or binary data"
        

        append_record(rec)
        ctx.log.info(f"[WS-DECOMP] WS {direction} host={host} flow={getattr(flow,'id',None)} decompressed={rec['decompressed']} text_len={(len(rec['message_text']) if rec['message_text'] else 0)}")
    except Exception as e:
        ctx.log.error(f"[WS-DECOMP] websocket_message main error: {e}")
        ctx.log.error(traceback.format_exc())




# # Expose module-level handlers (mitmproxy will pick these up)
addons = []

class _Shim:
    def websocket_handshake(self, flow):
        websocket_handshake(flow)
    def websocket_message(self, *a, **k):
        websocket_message(*a, **k)

addons.append(_Shim())
