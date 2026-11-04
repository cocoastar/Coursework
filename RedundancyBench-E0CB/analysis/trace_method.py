#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把完整轨迹自动拆成细事件，先事后逐事件初判，再逐事件做仅看当时信息的复核。

不读取人工 annotation.json；标注由规则提示 + 模型判断产生，不是 benchmark 原始人工标签。
仅从 GATEWAY_SK 读取密钥。上传个人聊天记录前请脱敏。结果可能含敏感信息。
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

GATEWAY_URL = os.environ.get("GATEWAY_URL", "https://matrixllm.alipay.com/v1/chat/completions")
MODEL = "gpt-6-luna"
LABELS = ("合理探索", "必要验证", "错误恢复", "低效重复", "正常推进", "证据不足")


def text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def canonical(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return value.strip()
    return text(value)


def tool_calls(msg):
    for call in msg.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or call
        yield str(fn.get("name", "unknown")), canonical(fn.get("arguments", {})), str(call.get("id") or "")


def load_traces(path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("simulations", [data])
    if not isinstance(data, list):
        raise ValueError("输入应为 final_traces.json、messages JSON 或 JSONL")
    for ordinal, rec in enumerate(data):
        if isinstance(rec, dict) and isinstance(rec.get("messages"), list):
            yield {"task_id": str(rec.get("task_id", rec.get("id", ordinal))),
                   "messages": rec["messages"]}


def compact_message(msg, idx, content_chars):
    raw = text(msg.get("content") or "")
    result = {"i": idx, "role": msg.get("role", "unknown"), "content": raw[:content_chars],
              "content_truncated": len(raw) > content_chars}
    calls = list(tool_calls(msg))
    if calls:
        result["tool_calls"] = [{"name": n, "arguments": a[:content_chars],
                                  "arguments_truncated": len(a) > content_chars, "id": cid}
                                 for n, a, cid in calls]
    if msg.get("role") == "tool":
        result.update(error=bool(msg.get("error")), requestor=msg.get("requestor"),
                      tool_call_id=msg.get("tool_call_id") or msg.get("id"))
    return result


def cost(messages, indices):
    """按原始消息去重计算；多工具调用共享一个消息时不会重复计算返回或消息数。"""
    sl = [messages[i] for i in sorted(set(indices))]
    returns = [text(m.get("content") or "") for m in sl if m.get("role") == "tool"]
    return {"assistant_tool_calls": sum(len(list(tool_calls(m))) for m in sl if m.get("role") == "assistant"),
            "user_tool_calls": sum(len(list(tool_calls(m))) for m in sl if m.get("role") == "user"),
            "other_tool_calls": sum(len(list(tool_calls(m))) for m in sl if m.get("role") not in ("user", "assistant")),
            "tool_returns": len(returns), "tool_return_chars": sum(map(len, returns)),
            "assistant_message_count": sum(m.get("role") == "assistant" for m in sl)}


def build_events(messages):
    """按调用 ID 匹配返回；同一个 ID 重用时按出现顺序匹配；不确定的返回不强行配对。"""
    events, pending, outstanding = [], defaultdict(deque), []
    covered = set()
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        calls = list(tool_calls(msg))
        if calls:
            for n, args, cid in calls:
                event = {"id": f"E{len(events) + 1:03d}", "kind": "tool_call", "role": role,
                         "start": i, "end": i, "message_indices": [i], "call_index": i,
                         "tool_name": n, "arguments": args, "call_id": cid,
                         "return_indices": [], "rule_hints": [], "related_event_ids": [],
                         "pairing": "awaiting_return"}
                events.append(event)
                if cid:
                    pending[cid].append(event)
                outstanding.append(event)
            covered.add(i)
        # 非空文字与工具调用同条消息时也是一个事件；成本按去重消息计。
        if role != "tool" and (not calls or text(msg.get("content") or "").strip()):
            events.append({"id": f"E{len(events) + 1:03d}", "kind": "text", "role": role,
                           "start": i, "end": i, "message_indices": [i], "return_indices": [],
                           "rule_hints": [], "related_event_ids": []})
            covered.add(i)
        if role == "tool":
            cid = str(msg.get("tool_call_id") or msg.get("id") or "")
            matched, pairing = None, "unpaired_return"
            if cid and pending[cid]:
                matched, pairing = pending[cid].popleft(), "id"
            elif not cid:
                adjacent = [e for e in outstanding if e["pairing"] == "awaiting_return" and e["call_index"] == i - 1]
                if len(adjacent) == 1:
                    matched, pairing = adjacent[0], "adjacent_no_id"
            if matched:
                matched["return_indices"].append(i)
                matched["message_indices"].append(i)
                matched["end"] = max(matched["end"], i)
                matched["pairing"] = pairing
                if matched in outstanding:
                    outstanding.remove(matched)
                if pairing == "adjacent_no_id":
                    matched["rule_hints"].append("返回无调用ID，按唯一相邻调用推测配对；需核对")
            else:
                events.append({"id": f"E{len(events) + 1:03d}", "kind": "unpaired_tool_return",
                               "role": "tool", "start": i, "end": i, "message_indices": [i],
                               "return_indices": [i], "pairing": pairing,
                               "rule_hints": ["工具返回无法可靠匹配调用；需核对"], "related_event_ids": []})
            covered.add(i)
        if i not in covered:
            events.append({"id": f"E{len(events) + 1:03d}", "kind": "text", "role": role,
                           "start": i, "end": i, "message_indices": [i], "return_indices": [],
                           "rule_hints": [], "related_event_ids": []})
            covered.add(i)
    # 并行调用可能返回顺序不同：按完成时间排序，id 仅是稳定标识，不代表时间顺序。
    events.sort(key=lambda e: (e["end"], e["start"], e["id"]))
    seen = {}
    for e in events:
        if e["kind"] == "tool_call":
            key = (e["tool_name"], e["arguments"])
            earlier = seen.get(key)
            if earlier:
                e["related_event_ids"].append({"event_id": earlier["id"], "relation": "same_tool_and_arguments"})
                e["rule_hints"].append("与之前调用的工具及参数相同；需核对目标、状态及返回是否改变")
                a = [text(messages[j].get("content") or "") for j in earlier["return_indices"]]
                b = [text(messages[j].get("content") or "") for j in e["return_indices"]]
                if a and b and a == b:
                    e["rule_hints"].append("相同参数且返回内容相同；这是检查线索，不是低效定论")
            if e["return_indices"] and any(messages[j].get("error") for j in e["return_indices"]):
                e["rule_hints"].append("工具返回报错；后续修复可能有价值")
            if e["pairing"] == "awaiting_return":
                e["rule_hints"].append("未找到对应工具返回；不能据此判为无进展")
            seen[key] = e
        e["cost"] = {"assistant_tool_calls": int(e["kind"] == "tool_call" and e["role"] == "assistant"),
                     "user_tool_calls": int(e["kind"] == "tool_call" and e["role"] == "user"),
                     "other_tool_calls": int(e["kind"] == "tool_call" and e["role"] not in ("assistant", "user")),
                     "tool_returns": len(e["return_indices"]),
                     "tool_return_chars": sum(len(text(messages[j].get("content") or "")) for j in e["return_indices"])}
        e["hindsight"] = None
        e["prefix"] = None
    return events


def parse_json(raw):
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        return json.loads(s[a:b + 1]) if a >= 0 and b > a else None


class GatewayHTTPError(RuntimeError):
    def __init__(self, status, detail):
        self.status = status
        super().__init__(f"HTTP {status}" + (f"：{detail}" if detail else "（响应中无可安全展示的错误细节）"))


def safe_gateway_detail(response):
    try:
        data = response.json()
    except (ValueError, TypeError):
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return ""
    parts = []
    for field in ("type", "code", "param"):
        v = error.get(field)
        if isinstance(v, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,60}", v):
            parts.append(f"{field}={v}")
    msg = error.get("message")
    if isinstance(msg, str) and any(t in msg.lower() for t in
                                    ("model", "parameter", "temperature", "completion", "context", "stream", "token")):
        msg = re.sub(r"sk-[a-zA-Z0-9._-]+", "[REDACTED]", msg.replace("\n", " ")[:220], flags=re.I)
        msg = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", "[REDACTED_EMAIL]", msg)
        msg = re.sub(r"(?<!\d)\d{11,}(?!\d)", "[REDACTED_NUMBER]", msg)
        parts.append("message=" + msg)
    return "; ".join(parts)


def call_model(prompt, *, max_tokens=6500, retries=2):
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("请先 pip install requests") from exc
    key = os.environ.get("GATEWAY_SK", "").strip()
    if not key:
        raise RuntimeError("请先自行设置环境变量 GATEWAY_SK；勿将密钥写入脚本或提交")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "stream": True, "max_completion_tokens": max_tokens}  # 不设置 temperature
    attempts = []
    for attempt in range(1, retries + 1):
        start, parts, usage = time.monotonic(), [], None
        try:
            with requests.post(GATEWAY_URL, headers=headers, json=body, timeout=(10, 180), stream=True) as resp:
                if resp.status_code != 200:
                    raise GatewayHTTPError(resp.status_code, safe_gateway_detail(resp))
                for raw_line in resp.iter_lines(decode_unicode=False):
                    line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
                    if not line or not line.startswith("data:"):
                        continue
                    s = line[5:].strip()
                    if s == "[DONE]":
                        break
                    try:
                        obj = json.loads(s)
                    except ValueError:
                        continue
                    usage = obj.get("usage") or usage
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            parts.append(delta["content"])
            raw = "".join(parts).strip()
            parsed = parse_json(raw)
            if not isinstance(parsed, dict):
                raise ValueError("模型返回未能解析为 JSON 对象")
            attempts.append({"elapsed_s": round(time.monotonic() - start, 3), "usage": usage, "ok": True})
            return parsed, {"model": MODEL, "attempts": attempts, "raw": raw}
        except Exception as exc:
            err = f"{type(exc).__name__}: " + (str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "网络/超时故障")
            attempts.append({"elapsed_s": round(time.monotonic() - start, 3), "usage": usage,
                             "ok": False, "error": err})
            if isinstance(exc, GatewayHTTPError) and 400 <= exc.status < 500 and exc.status != 429:
                break
            if attempt < retries:
                time.sleep(attempt)
    return None, {"model": MODEL, "attempts": attempts, "raw": "",
                  "error": "模型调用失败：" + " | ".join(a["error"] for a in attempts)}


SCAN_PROMPT = """你在给完整 agent 轨迹做事后分析。轨迹按每次工具调用及其返回、独立文字消息拆成细事件；
全部消息按原始下标给出，可能有多事件共享一条消息。仅逐个评价 target_events 中的事件，其他消息仅是背景。
这是新的模型标注，不是 benchmark 人工原标签。不要因工具/参数不同就称冗余；相同也不能只凭相同判浪费。
请识别目标、对象、进展或新增信息，并对比以前行动。回退必须有操作→撤销→再操作的具体证据。
失败的尝试如果当时在排除合理选项、获得新信息、核验风险或修复故障，可能有价值；不要用最后结果倒推无效。
类别只选：合理探索、必要验证、错误恢复、低效重复、正常推进、证据不足。
低效重复须具体指出以前已经知道什么、再做为什么没有新收益；不确定时用证据不足。独立用户要求通常属正常推进。
每个目标输出 id,label,evidence_indices(原始消息下标，可引用此前相关步骤),reason(通俗说明),
objective,object_or_target,new_information_or_risk,value_if_failed(bool),needs_review(bool),backtrack_evidence(无则空字符串)。
如同一调用本身难分，标证据不足并 needs_review=true。不因规则提示就贴低效标签。
严格 JSON: {"events":[...]}; target_events 每一个 id 都要出现且只出现一次。
输入：\n"""
PREFIX_PROMPT = """你正在轨迹的当前时刻，只能看 messages_so_far：绝不能利用未来结果判断 target。
轨迹按小事件拆分，本次只判断 target 这个事件，允许引用以前消息的下标作为证据。
类别只选：合理探索、必要验证、错误恢复、低效重复、正常推进、证据不足。
有明确探索/验证/故障恢复价值的失败不等于低效。低效要解释为何当时已有充分信息、状态目标未变、再做无新收益；不确定选证据不足。
工具响应可能迟到或没有配对，不要凭相同工具名或最终没成功就判浪费。
严格 JSON: {"label":"六类之一","evidence_indices":[原始整数下标],"reason":"通俗说明",
"what_was_known":"当时已知的事实","new_information_or_risk":"新信息或需核验的风险",
"would_skip_be_safe":"是/否/不确定","valuable_despite_failure":false,"needs_review":false,
"backtrack_evidence":"无则空字符串"}。
输入：\n"""


def normalize_judgment(obj):
    if obj.get("label") == "验证":
        obj["label_original"] = obj["label"]
        obj["label"] = "必要验证"
        obj["needs_review"] = True
    return obj


def evidence_ok(item, event, last_index):
    idx = item.get("evidence_indices")
    return (isinstance(idx, list) and all(type(x) is int and 0 <= x <= last_index for x in idx)
            and (item.get("label") != "低效重复" or
                 (any(x in event["message_indices"] for x in idx)
                  and any(x < event["start"] for x in idx))))



def candidate_stages(events, messages):
    """只把连续多步、两种视角都认为低效的段落提出检查，不当作已证实无进展。"""
    stages, run = [], []
    def emit():
        if len(run) < 3:
            return
        stages.append({"event_ids": [e["id"] for e in run],
                       "start": min(e["start"] for e in run), "end": max(e["end"] for e in run),
                       "cost": cost(messages, (j for e in run for j in e["message_indices"])),
                       "reason": "连续至少三个小事件，事后与当时视角均认为值得检查低效重复；仍需人工看用户目标/状态是否变化"})
    for e in events:
        past, now = e.get("hindsight"), e.get("prefix")
        if (past and now and past.get("label") == now.get("label") == "低效重复"
                and not past.get("value_if_failed") and not now.get("valuable_despite_failure")):
            run.append(e)
        else:
            emit()
            run = []
    emit()
    return stages

def process(rec, args):
    messages = rec["messages"]
    if not messages:
        raise ValueError("空轨迹")
    events = build_events(messages)
    visible = [compact_message(m, i, args.content_chars) for i, m in enumerate(messages)]
    total = cost(messages, range(len(messages)))
    record = {"task_id": rec["task_id"], "message_count": len(messages),
              "event_count": len(events), "truncated_message_count": sum(m["content_truncated"] for m in visible),
              "total_cost": total, "events": events, "reviewed_prefix_ids": [],
              "not_prefix_reviewed_ids": [], "scan_missing_ids": [], "manual_review_ids": [],
              "model_calls": [], "status": "dry_run" if args.dry_run else "partial",
              "suspected_stagnation_stages": [], "backtrack_candidate_ids": [],
              "caveat": "本结果是新模型判断，不是原始人工标签；缺少可靠轨迹 token/耗时则仅报告调用数及字符数（字符不是 token）；模型 usage/请求耗时属于标注开销。规则提示不是冗余定论。"}
    assert set(range(len(messages))) == {j for e in events for j in e["message_indices"]}
    assert total["assistant_tool_calls"] == sum(e["cost"]["assistant_tool_calls"] for e in events)
    if args.dry_run:
        return record
    by_id = {e["id"]: e for e in events}
    for offset in range(0, len(events), args.scan_batch_events):
        batch = events[offset:offset + args.scan_batch_events]
        timeline = {"task_id": rec["task_id"], "messages": visible,
                    "target_events": [{k: e[k] for k in ("id", "kind", "role", "start", "end", "message_indices",
                                                      "tool_name", "return_indices", "rule_hints", "related_event_ids")
                                       if k in e} | ({"arguments": e["arguments"][:args.content_chars],
                                                      "arguments_truncated": len(e["arguments"]) > args.content_chars}
                                                     if "arguments" in e else {}) for e in batch]}
        prompt = SCAN_PROMPT + json.dumps(timeline, ensure_ascii=False)
        if len(prompt) > args.max_prompt_chars:
            for e in batch:
                e["hindsight_error"] = f"事后提示 {len(prompt)} 字符超上限；未静默截断"
            continue
        obj, meta = call_model(prompt, max_tokens=args.max_completion_tokens)
        record["model_calls"].append({"stage": "hindsight_batch", "event_ids": [e["id"] for e in batch], **meta})
        if obj is None:
            for e in batch:
                e["hindsight_error"] = meta["error"]
            continue
        target = {e["id"] for e in batch}
        seen = set()
        for item in obj.get("events", []) if isinstance(obj.get("events"), list) else []:
            if not isinstance(item, dict) or item.get("id") not in target or item["id"] in seen:
                continue
            e = by_id[item["id"]]
            seen.add(item["id"])
            if normalize_judgment(item).get("label") in LABELS and evidence_ok(item, e, len(messages) - 1):
                e["hindsight"] = item
            else:
                e["hindsight_error"] = "标签/证据下标非法或低效缺少前因证据；检查 raw"
    record["scan_missing_ids"] = [e["id"] for e in events if e["hindsight"] is None]
    chronological = sorted(events, key=lambda e: (e["end"], e["start"], e["id"]))
    selected = chronological if args.max_prefix == 0 else chronological[:args.max_prefix]
    record["not_prefix_reviewed_ids"] = [e["id"] for e in chronological[len(selected):]]
    for e in selected:
        prefix = {"task_id": rec["task_id"], "target": {k: e[k] for k in
                  ("id", "kind", "role", "start", "end", "message_indices", "rule_hints", "related_event_ids")},
                  "messages_so_far": visible[:e["end"] + 1]}
        prompt = PREFIX_PROMPT + json.dumps(prefix, ensure_ascii=False)
        if len(prompt) > args.max_prompt_chars:
            e["prefix_error"] = f"前缀提示 {len(prompt)} 字符超上限；未发送"
            continue
        obj, meta = call_model(prompt, max_tokens=args.max_completion_tokens)
        record["model_calls"].append({"stage": "prefix", "event_id": e["id"], **meta})
        if obj is None:
            e["prefix_error"] = meta["error"]
        elif normalize_judgment(obj).get("label") in LABELS and evidence_ok(obj, e, e["end"]):
            e["prefix"] = obj
            record["reviewed_prefix_ids"].append(e["id"])
        else:
            e["prefix_error"] = "标签/证据下标非法或低效缺少前因证据；检查 raw"
    record["manual_review_ids"] = [e["id"] for e in events if
        e.get("hindsight_error") or e.get("prefix_error") or e["rule_hints"]
        or e["id"] in record["not_prefix_reviewed_ids"]
        or (e["hindsight"] and (e["hindsight"].get("needs_review") or e["hindsight"].get("value_if_failed")
            or e["hindsight"].get("label") == "证据不足"))
        or (e["prefix"] and (e["prefix"].get("needs_review") or e["prefix"].get("valuable_despite_failure")
            or e["prefix"].get("label") == "证据不足"
            or (e["hindsight"] and e["hindsight"].get("label") != e["prefix"].get("label"))))]
    record["suspected_stagnation_stages"] = candidate_stages(chronological, messages)
    no_evidence = {"", "无", "无证据", "没有", "无明确证据", "none", "n/a"}
    record["backtrack_candidate_ids"] = [e["id"] for e in events if any(
        isinstance(view, dict) and str(view.get("backtrack_evidence") or "").strip().lower() not in no_evidence
        for view in (e["hindsight"], e["prefix"]))]
    record["status"] = "ok" if not record["scan_missing_ids"] and len(record["reviewed_prefix_ids"]) == len(events) else "partial"
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="final_traces.json / 自己导出的 messages JSON / JSONL")
    ap.add_argument("--task-id", help="精确匹配 task_id")
    ap.add_argument("--limit", type=int, default=2, help="最多处理几条，0 为全部")
    ap.add_argument("--output", type=Path, default=Path("trace_judgments.jsonl"))
    ap.add_argument("--overwrite", action="store_true", help="显式覆盖已有结果")
    ap.add_argument("--dry-run", action="store_true", help="只拆事件/规则提示，不调用模型")
    ap.add_argument("--window-messages", type=int, help="旧参数兼容：新版本改为事件划分，此参数不起作用")
    ap.add_argument("--scan-batch-events", type=int, default=8, help="每次事后模型判断的细事件数，默认 8")
    ap.add_argument("--content-chars", type=int, default=1600, help="给模型的单消息最多字符数，截断会标记")
    ap.add_argument("--max-prefix", type=int, default=0, help="前缀复核事件数量；默认 0=全部，设非零则状态 partial")
    ap.add_argument("--max-prompt-chars", type=int, default=150000, help="提示过长直接报错，不静默截断")
    ap.add_argument("--max-completion-tokens", type=int, default=6500)
    args = ap.parse_args()
    if args.limit < 0 or args.max_prefix < 0 or args.scan_batch_events < 1 or args.content_chars < 100:
        ap.error("limit/max-prefix 不可为负，scan-batch-events 至少 1，content-chars 至少 100")
    if not args.input.exists():
        ap.error(f"输入文件不存在：{args.input}")
    if args.output.resolve() == args.input.resolve():
        ap.error("输出文件不得与输入文件相同")
    if args.output.exists() and not args.overwrite:
        ap.error(f"输出文件已存在：{args.output}；更换 --output 或加 --overwrite")
    if not args.dry_run and not os.environ.get("GATEWAY_SK", "").strip():
        ap.error("缺少 GATEWAY_SK；--dry-run 不需要密钥")
    if args.window_messages is not None:
        print("提示：--window-messages 已废弃；现在按调用/返回细分事件", file=sys.stderr)
    recs = (r for r in load_traces(args.input) if not args.task_id or r["task_id"] == args.task_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", encoding="utf-8") as out:
        os.chmod(args.output, 0o600)
        for rec in recs:
            if args.limit and n >= args.limit:
                break
            result = process(rec, args)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            n += 1
            print(f"[{n}] task_id={rec['task_id']} messages={result['message_count']} "
                  f"events={len(result['events'])} model_calls={len(result['model_calls'])} "
                  f"status={result['status']}", file=sys.stderr, flush=True)
    if n == 0:
        print("未匹配轨迹：检查 --task-id", file=sys.stderr)
        return 1
    print(f"完成：{args.output}；模型请求/usage/耗时见 model_calls（重试另计）。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
