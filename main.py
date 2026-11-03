# -*- coding: utf-8 -*-
"""
对话 Token 用量 · Token Meter  v1.2
====================================

【v1.2 修了什么】

v1.1 有两条路各算各的：/查询token 用估算，自动提醒用 provider_stats 的真实值。
结果两个数字打起来了：查询说 30.2%，提醒说 308.0%。

308% 的来路：provider_stats 里有一条 2026-10-27 14:17 的记录，
cached 字段写着 1,933,952（超过模型窗口，明显是脏数据），
提醒就把它当真了。

v1.2 改成**同一个口径**：查询和提醒都走 _scan_session()，
一次扫描把总数、条数、角色分布一起算出来。两个数字永远一致。

provider_stats 的真实值还留着，但只在「查询」里做参照显示，
不参与判断。

【它干什么】

一，查

    /查询token

    对话 Token 用量

    已用  19.0 万 / 62.9 万
    占比  30.2%
    [██████░░░░░░░░░░░░░░]

    消息  305 条（你 90 / 我 152 / 工具 63）
    你的话占  6.4 万
    其中思考  3.6 万
    上次请求实测  22.4 万 输入（deepseek-flash）

二，到线自动提醒

    每一次回复发出之前，看一眼用量。超过阈值（默认 95%）就在回复末尾
    附一句（单行，不会再被分段拆成四个气泡）。
    提醒过之后要再涨 `remind_cooldown` 个 token 才会再提，免得每句话都念。

【怎么算的】

照 AstrBot 自己的估算规则（astrbot/core/agent/context/token_counter.py）：

    中文        0.6 token / 字
    其它字符    0.3 token / 字（英文、数字、标点）
    emoji       2.0 token / 个
    图片        765 token / 张
    音频        500 token / 段
    工具调用    把参数 JSON 当文字，按上面同样规则算

为什么图片不是按字符算：图片在库里是 base64，一张截图能有 40 万字符。
按字符算会得出「这个对话有几十亿 token」这种荒唐结果。
模型看图片只花几百 token，所以按张算。

【上限填多少】

这个数字只用来算百分比，不是模型窗口。

想让百分比等于「离压缩还有多远」，就填压缩线：

    压缩线 = 模型上下文窗口 × 0.82（这个 0.82 写死在 AstrBot 里）

例：flash 窗口填 767000，压缩线就是 767000 × 0.82 = 628940。

【安全】

  1. 查 token 只允许管理员
  2. 全程只读数据库（mode=ro，读不到就退 immutable=1）
  3. 不写任何文件，不改任何配置
  4. 自动提醒只往回复末尾追加一行文字，不接管发送流程
"""

import json
import re
import sqlite3
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star

# ---- 与 AstrBot 内部 EstimateTokenCounter 保持一致的估算规则 ----
TOK_CHINESE = 0.6
TOK_OTHER = 0.3
TOK_EMOJI = 2.0
TOK_IMAGE = 765
TOK_AUDIO = 500

# emoji 区段（含 ZWJ 与变体选择符，撑旗子、多人 emoji 要靠它们）
EMOJI_PATTERN = re.compile(
    "[\u200d\u2600-\u27bf\ufe00-\ufe0f\U0001f000-\U0001faff]"
)
# 用正则数汉字，比逐字符循环快好几倍（库大了以后差别明显）
CJK_PATTERN = re.compile("[\u4e00-\u9fff]")


def _estimate(text) -> int:
    """估算一段文字的 token 数。"""
    if not text or not isinstance(text, str):
        return 0
    cn = len(CJK_PATTERN.findall(text))
    em = len(EMOJI_PATTERN.findall(text))
    other = len(text) - cn - em
    if other < 0:
        other = 0
    return int(cn * TOK_CHINESE + em * TOK_EMOJI + other * TOK_OTHER)


def _count_message(msg: dict, count_think: bool = True) -> int:
    """估算单条消息的 token 数（含工具调用参数）。"""
    if not isinstance(msg, dict):
        return 0
    total = 0

    content = msg.get("content")
    if isinstance(content, str):
        total += _estimate(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                total += _estimate(part)
                continue
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                total += _estimate(part.get("text") or "")
            elif ptype == "think":
                if count_think:
                    total += _estimate(part.get("think") or "")
            elif ptype in ("image_url", "image"):
                total += TOK_IMAGE
            elif ptype in ("audio_url", "audio"):
                total += TOK_AUDIO
            else:
                # 不认识的类型，按 JSON 文字长度估，宁可偏高
                total += _estimate(json.dumps(part, ensure_ascii=False))

    tcs = msg.get("tool_calls")
    if tcs:
        try:
            total += _estimate(json.dumps(tcs, ensure_ascii=False))
        except Exception:
            pass

    return total


class TokenMeterPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}

        # main.py -> 插件目录 -> plugins -> data
        self.data_dir = Path(__file__).resolve().parent.parent.parent
        self.db_path = self.data_dir / "data_v4.db"

        # 上一次提醒时的 token 数，用来做冷却
        self._last_reminded = 0

        logger.info(f"[Token用量] 数据库路径: {self.db_path}")
        logger.info(
            "[Token用量] v1.2 自动提醒: %s（阈值 %s%%，冷却 %s token）"
            % (
                "开" if self.config.get("remind_enable", True) else "关",
                self.config.get("remind_percent", 95.0),
                self.config.get("remind_cooldown", 5000),
            )
        )

    # ---------------- 数据库（只读） ----------------

    def _uri_base(self) -> str:
        return "file:" + str(self.db_path).replace("\\", "/")

    def _fetch_rows(self):
        """读 conversations。运行中的库带 WAL，先试 mode=ro，
        被锁就退到 immutable=1（可能读不到最后几分钟，但不影响看总量）。"""
        last_err = None
        for qs in ("?mode=ro", "?mode=ro&immutable=1"):
            con = None
            try:
                con = sqlite3.connect(self._uri_base() + qs, uri=True, timeout=3)
                return con.execute(
                    "SELECT user_id, platform_id, title, content FROM conversations"
                ).fetchall()
            except sqlite3.Error as e:
                last_err = e
            finally:
                if con is not None:
                    try:
                        con.close()
                    except Exception:
                        pass
        raise last_err or RuntimeError("读数据库失败")

    def _pick_conversation(self, rows, umo: str = ""):
        """选会话：先按 UMO 精确匹配，再按尾部宽松匹配，最后取内容最长的。"""
        if not rows:
            return None

        if umo:
            for r in rows:
                if r[0] == umo:
                    return r
            tail = umo.rsplit(":", 1)[-1] if ":" in umo else umo
            if tail:
                for r in rows:
                    if r[0] and r[0].endswith(tail):
                        return r

        return max(rows, key=lambda r: len(r[3] or ""))

    def _real_usage(self, umo: str):
        """取最近一次真实 API 调用的输入 token（provider_stats）。

        ⚠ v1.1 这里出过事：agent_type=internal 的记录里混着远超模型窗口的
        脏数据（有一条 cached=1,933,952），直接拿来算百分比就会得出 308%。
        现在只拿来当参照显示，不参与任何判断；并且在这里先做一遍合理性筛。

        返回 None 表示没有可信数据。
        """
        sql = (
            "SELECT umo, token_input_other, token_input_cached, token_output, provider_model "
            "FROM provider_stats WHERE status='completed' ORDER BY id DESC LIMIT 400"
        )
        rows = None
        for qs in ("?mode=ro", "?mode=ro&immutable=1"):
            con = None
            try:
                con = sqlite3.connect(self._uri_base() + qs, uri=True, timeout=3)
                rows = con.execute(sql).fetchall()
                break
            except sqlite3.Error as e:
                logger.warning(f"[Token用量] 读 provider_stats 失败：{e}")
            finally:
                if con is not None:
                    try:
                        con.close()
                    except Exception:
                        pass
        if not rows:
            return None

        # 只认这条会话自己的记录，不做全局兜底（兜底会拿到别人的数）
        picked = None
        if umo:
            for r in rows:
                if r[0] == umo:
                    picked = r
                    break
        if picked is None:
            return None

        total = int(picked[1] or 0) + int(picked[2] or 0)

        # 合理性：单次请求的输入不可能超过模型窗口的量级。
        # 超过上限的，是脏数据，不要。
        try:
            max_tokens = int(self.config.get("max_tokens", 1000000) or 1000000)
        except Exception:
            max_tokens = 1000000
        if max_tokens <= 0:
            max_tokens = 1000000
        if total <= 0 or total > max_tokens * 2:
            return None

        return {
            "input": total,
            "other": int(picked[1] or 0),
            "cached": int(picked[2] or 0),
            "output": int(picked[3] or 0),
            "model": picked[4] or "",
        }

    # ---------------- 唯一的统计口径 ----------------

    def _scan_session(self, umo: str):
        """把整个会话扫一遍。

        查询和提醒都用这一个结果，所以两个数字永远一致。

        返回 dict：total / count / user_tok / think_tok / roles / ok / err
        """
        out = {
            "ok": False,
            "err": "",
            "total": 0,
            "count": 0,
            "user_tok": 0,
            "think_tok": 0,
            "roles": {},
            "uid": "",
        }

        if not self.db_path.exists():
            out["err"] = f"找不到数据库：{self.db_path}"
            return out

        try:
            rows = self._fetch_rows()
        except Exception as e:
            out["err"] = f"读数据库失败：{e}"
            return out

        conv = self._pick_conversation(rows, umo)
        if not conv:
            out["err"] = "数据库里没有会话记录。"
            return out

        uid, _platform_id, _title, content = conv
        out["uid"] = uid or ""

        try:
            msgs = json.loads(content) if content else []
        except Exception as e:
            out["err"] = f"解析会话内容失败：{e}"
            return out
        if not isinstance(msgs, list):
            out["err"] = "会话内容不是消息列表，无法统计。"
            return out

        count_think = bool(self.config.get("count_think", True))
        total = 0
        user_tok = 0
        think_tok = 0
        roles = {}

        for m in msgs:
            t = _count_message(m, count_think)
            total += t

            role = (m.get("role") if isinstance(m, dict) else None) or "?"
            roles[role] = roles.get(role, 0) + 1
            if role == "user":
                user_tok += t

            if isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "think":
                            think_tok += _estimate(part.get("think") or "")

        out.update(
            {
                "ok": True,
                "total": total,
                "count": len(msgs),
                "user_tok": user_tok,
                "think_tok": think_tok,
                "roles": roles,
            }
        )
        return out

    # ---------------- 显示 ----------------

    @staticmethod
    def _fmt(n: int) -> str:
        if n >= 10000:
            return "%.1f 万" % (n / 10000.0)
        return "%d" % int(n)

    def _limits(self):
        """取 max_tokens / 提醒阈值 / 冷却，统一处理非法值。"""
        try:
            max_tokens = int(self.config.get("max_tokens", 1000000) or 1000000)
        except Exception:
            max_tokens = 1000000
        if max_tokens <= 0:
            max_tokens = 1000000

        try:
            pct_line = float(self.config.get("remind_percent", 95.0))
        except Exception:
            pct_line = 95.0
        if pct_line <= 0 or pct_line > 100:
            pct_line = 95.0

        try:
            cooldown = int(self.config.get("remind_cooldown", 5000))
        except Exception:
            cooldown = 5000
        if cooldown < 0:
            cooldown = 0

        return max_tokens, pct_line, cooldown

    def _build_report(self, event: AstrMessageEvent) -> str:
        umo = getattr(event, "unified_msg_origin", "") or ""
        s = self._scan_session(umo)
        if not s["ok"]:
            return s["err"]

        used = s["total"]
        msgs_count = s["count"]
        roles = s["roles"]
        user_tok = s["user_tok"]
        think_tok = s["think_tok"]

        max_tokens, _pct_line, _cooldown = self._limits()

        ratio = used / float(max_tokens)
        pct = ratio * 100.0

        lines = ["对话 Token 用量", ""]
        lines.append("已用  %s / %s" % (self._fmt(used), self._fmt(max_tokens)))
        lines.append("占比  %.1f%%" % pct)

        if self.config.get("show_bar", True):
            filled = int(round(min(max(ratio, 0.0), 1.0) * 20))
            lines.append("[" + "█" * filled + "░" * (20 - filled) + "]")

        if self.config.get("show_detail", True):
            lines.append("")
            lines.append(
                "消息  %d 条（你 %d / 我 %d / 工具 %d）"
                % (
                    msgs_count,
                    roles.get("user", 0),
                    roles.get("assistant", 0),
                    roles.get("tool", 0),
                )
            )
            lines.append("你的话占  %s" % self._fmt(user_tok))
            if bool(self.config.get("count_think", True)) and think_tok:
                lines.append("其中思考  %s" % self._fmt(think_tok))

            real = self._real_usage(s["uid"])
            if real:
                lines.append(
                    "上次请求实测  %s 输入（%s）"
                    % (self._fmt(real["input"]), real["model"] or "?")
                )

        try:
            warn = float(self.config.get("warn_percent", 80.0) or 80.0)
        except Exception:
            warn = 80.0
        if pct >= 100.0:
            lines.append("")
            lines.append("已经越过上限了")
        elif pct >= warn:
            lines.append("")
            lines.append("快到头了（超过 %g%%）" % warn)

        if self.config.get("verbose", False):
            logger.info(
                f"[Token用量] uid={s['uid']} msgs={msgs_count} used={used} max={max_tokens}"
            )

        return "\n".join(lines)

    # ---------------- 自动提醒 ----------------

    def _should_remind(self, event: AstrMessageEvent):
        """要提醒就返回一行文字，不要就返回 None。"""
        if not self.config.get("remind_enable", True):
            return None

        umo = getattr(event, "unified_msg_origin", "") or ""

        # 默认只在私聊提醒，群聊里蹦出这种数字不合适
        if self.config.get("remind_private_only", True):
            if "FriendMessage" not in umo:
                return None

        # 指定会话（留空 = 所有私聊）
        target = str(self.config.get("remind_target", "") or "").strip()
        if target and target != umo:
            return None

        # 命令消息不提醒（/查询token 的回复下面再挂一句就是吵）
        msg = (event.message_str or "").strip()
        if msg.startswith("/") or msg.startswith("／"):
            return None

        s = self._scan_session(umo)
        if not s["ok"]:
            return None

        used = s["total"]
        if used <= 0:
            return None

        max_tokens, pct_line, cooldown = self._limits()
        pct = used * 100.0 / max_tokens

        if pct < pct_line:
            return None

        # 冷却：已经提醒过，且涨得不够多，就不再念
        if self._last_reminded > 0 and (used - self._last_reminded) < cooldown:
            return None

        self._last_reminded = used

        left = max_tokens - used
        if left > 0:
            text = "上下文用量到 %.1f%% 了，还剩 %s token，到 100%% 会自动压缩" % (
                pct,
                self._fmt(left),
            )
        else:
            text = "上下文用量 %.1f%%，已过压缩线" % pct

        if self.config.get("verbose", False):
            logger.info(f"[Token用量] 触发提醒：{used}/{max_tokens} = {pct:.1f}%")

        return text

    @filter.on_decorating_result()
    async def remind_before_send(self, event: AstrMessageEvent):
        """在回复发出之前看一眼用量，到线就在末尾附一句。

        只往消息链尾追加**一行**文字，不接管发送流程。
        v1.1 那版追加的是多行，被平台的自动分段拆成了四个气泡，
        看着像刷屏，所以改单行。
        """
        try:
            extra = self._should_remind(event)
            if not extra:
                return

            result = event.get_result()
            if not result or not result.chain:
                return

            result.chain.append(Plain(extra))
        except Exception as e:
            logger.warning(f"[Token用量] 提醒钩子出错：{e}")

    # ---------------- 指令 ----------------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("查询token")
    async def cmd_query_token(self, event: AstrMessageEvent):
        """查看当前对话的 token 用量"""
        try:
            text = self._build_report(event)
        except Exception as e:
            logger.error(f"[Token用量] 查询失败：{e}")
            yield event.plain_result(f"查 token 失败：{e}")
            return
        yield event.plain_result(text)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("token")
    async def cmd_token(self, event: AstrMessageEvent):
        """英文别名"""
        try:
            text = self._build_report(event)
        except Exception as e:
            logger.error(f"[Token用量] 查询失败：{e}")
            yield event.plain_result(f"查 token 失败：{e}")
            return
        yield event.plain_result(text)
