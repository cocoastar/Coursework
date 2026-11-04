#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成本较低的轨迹标注 v2：全量细事件初筛 + 少量真正“调用前”复核。

只引用不同 task 的人工示例，不读取 annotation.json 作为当前任务标签；
原版 trace_method.py 保留，便于比较。密钥仍仅来自 GATEWAY_SK。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from trace_method import (MODEL, LABELS, build_events, call_model, compact_message,
                          cost, load_traces, normalize_judgment, text)

# 经过原始轨迹核对的匿名化人工作业案例，仅用于解释判别边界；task_id 相同的案例绝不喂给模型。
# 不含具体用户身份、原文轨迹或被评价轨迹的标签。可用 --no-reference-cases 完全关闭。
REFERENCE_CASES = (
    {"source_task_id": "5", "source_indices": [6, 7], "original_tag": "Exploratory Step",
     "lesson": "用户未给目标预订号时逐个查候选；虽查到不匹配航班、后续未直接用到，这次查询排除了一个候选。事前探索有理、事后没有进入最终写入，两件事要分开。"},
    {"source_task_id": "5", "source_indices": [16, 17], "original_tag": "Incorrect Step",
     "lesson": "所需目标预订已定位后，又查无关预订，没有解决剩余问题，也没参与处理；不要将所有查询都自动叫必要验证。"},
    {"source_task_id": "14", "source_indices": [12, 13], "original_tag": "Duplicated Step",
     "lesson": "紧接着同工具、同参数查询，前次已有有效结果，本次返回也相同，无目标/状态变化；是应进一步检查的重复，不代表所有相同工具调用都重复。"},
)
EX_ANTE = {"有依据", "缺少依据", "不确定"}
OUTCOME = {"新增信息", "排除候选", "状态改变", "重复已有", "无法判断"}
USED = {"是", "否", "不确定"}

SCAN = """你在研究完整轨迹，必须对 target_events 每个小事件分别判断，不能只评可疑段。
参考案例是不同任务的人工标注总结，不是本任务的答案；原标签中的 Exploratory 有可能有事前价值但未进入最终处理。
请分开判断两件事：(1) 在行动前，当时是否有理由；(2) 返回后有没有新增信息、排除候选、状态变化，及是否被后续真正使用。
同工具不同参数不能算重复；同参数也要考虑状态变化、失败重试和新用户要求。不能只靠最后失败/没用上就判浪费。
“必要验证”仅用于若不核验就有明确且具体风险、没有其他途径可安全完成的检查；只是‘可查库存/价格’但不是必需，至多合理探索/证据不足。
任务已确认目标后仍查无关对象，可能低效，但要给出目标和上下文证据。普通执行操作选正常推进。
错误恢复要有实际报错，尝试排错有价值；本次没看到报错不能编造。长时间停滞与回退需有多步证据，不能仅凭耗时或工具数。
只输出 JSON {"events":[{"id":"E...","label":"合理探索/必要验证/错误恢复/低效重复/正常推进/证据不足",
"ex_ante_reason":"有依据/缺少依据/不确定","outcome":"新增信息/排除候选/状态改变/重复已有/无法判断",
"used_later":"是/否/不确定","observed_failure":false,"valuable_if_failed":false,
"evidence_indices":[原始消息下标整数],"backtrack_from_indices":[仅操作→撤销→再操作才引用的较早原始下标],"reason":"一句话，不得夸大必要性","needs_review":false}]}。
每个目标 id 恰好一次。低效重复需指出目标/参数/状态没变以及较早哪条消息已经足够；证据不足则标待复核。
输入：\n"""

BEFORE = """请仅评价 proposed_action 在调用前是否值得执行：历史严格止于行动之前，尚不知道这次行动的返回内容。
候选和事后意见仅指明要审查哪一步，不能当成事实；只根据 messages_before_action 和 proposed_action 判断。
同一次候选查询即使未来未用上，若能排除合理选项仍可有事前价值；库存查询若无明确必须核验的风险，不能自动称为必要验证。
若之前的相同请求已给出有效结果、目标状态未变，应检查是否无理由重做；报错或状态变更则需另论。
仅输出 JSON {"label":"合理探索/必要验证/错误恢复/低效重复/正常推进/证据不足",
"ex_ante_reason":"有依据/缺少依据/不确定","would_skip_be_safe":"是/否/不确定",
"evidence_indices":[只能引行动之前的原始下标整数],"reason":"一句话", "needs_review":false}。
注意：proposed_action 是要判断的当前动作，不需在 evidence_indices 中重复填当前下标。
输入：\n"""


def references(task_id, enabled):
    return [{"source": f"airline task {x['source_task_id']} {x['source_indices']}",
             "original_tag": x["original_tag"], "lesson": x["lesson"]}
            for x in REFERENCE_CASES if enabled and x["source_task_id"] != str(task_id)]


def basic_evidence(item, max_index):
    indices = item.get("evidence_indices")
    return (isinstance(indices, list) and all(type(i) is int and 0 <= i <= max_index for i in indices))


def validate_scan(item, event, last_index):
    if item.get("label") not in LABELS or not basic_evidence(item, last_index):
        return "标签或证据下标非法"
    warnings = []
    for key, options in (("ex_ante_reason", EX_ANTE), ("outcome", OUTCOME), ("used_later", USED)):
        if item.get(key) not in options:
            item[key] = "不确定" if key != "outcome" else "无法判断"
            warnings.append(key + " 缺失/非法，已标记不确定")
    if item["label"] == "低效重复":
        idx = item["evidence_indices"]
        if not any(i < event["start"] for i in idx):
            warnings.append("低效判断缺少较早的证据")
        if not any(i in event["message_indices"] for i in idx):
            warnings.append("证据数组漏写当前动作；原始输出保留，未伪造模型引用")
    if warnings:
        item["needs_review"] = True
        item["evidence_warning"] = warnings
    return None


def choose_targets(events, max_prefix, controls):
    """选候选而非给正常段自动贴好标签；留一个对照样本，未抽中的候选显式列出。"""
    ranked = []
    for e in events:
        h = e["hindsight"] or {}
        if not h:
            score, why = 120, "初筛缺失"
        elif e["rule_hints"]:
            score, why = 110, "规则提示需查状态/前因"
        elif h.get("label") in ("低效重复", "证据不足"):
            score, why = 100, "初筛认为低效或证据不足"
        elif h.get("ex_ante_reason") != "有依据":
            score, why = 80, "行动前理由有疑问"
        elif h.get("outcome") in ("重复已有", "无法判断") or h.get("used_later") == "否":
            score, why = 65, "结果贡献/使用情况待查"
        elif e["kind"] == "tool_call" and any(w in e.get("tool_name", "").lower() for w in ("search", "list", "query")):
            score, why = 45, "搜索/列举步骤：不应自动判必要"
        elif h.get("needs_review") or h.get("observed_failure") or h.get("valuable_if_failed"):
            score, why = 35, "失败价值/模型建议复核"
        else:
            continue
        e["selection_reason"] = why
        # 在同一优先级内优先看工具返回较长者；但仍有对照样本。
        ranked.append((score, min(e["cost"]["tool_return_chars"], 3000), -e["start"], e))
    ranked.sort(key=lambda row: (-row[0], -row[1], -row[2]))
    candidates = [row[3] for row in ranked]
    other = [e for e in events if e not in candidates and e["kind"] == "tool_call"]
    other.sort(key=lambda e: (e["hindsight"] or {}).get("outcome") != "排除候选")
    control_items = other[:controls]
    for e in control_items:
        e["selection_reason"] = "正常段对照抽样"
    if max_prefix:
        # 优先给少量对照留席位，避免只在可疑段检验模型的误报。
        reserve = min(len(control_items), max_prefix)
        selected = candidates[:max_prefix - reserve] + control_items[:reserve]
    else:
        selected = candidates + control_items
    selected_ids = {e["id"] for e in selected}
    return selected, [e["id"] for e in candidates if e["id"] not in selected_ids], [e["id"] for e in control_items]


def scan_stage_candidates(events, messages):
    """便宜的连续停滞检查线索：模型逐步结果，绝不声称轨迹真实耗时或已证实浪费。"""
    found, run = [], []
    def emit():
        if len(run) >= 3:
            found.append({"event_ids": [e["id"] for e in run],
                          "message_start": min(e["start"] for e in run),
                          "message_end": max(e["end"] for e in run),
                          "observable_cost": cost(messages, (i for e in run for i in e["message_indices"])),
                          "reason": "连续至少3个事件初筛认为没有明确新信息；仍须检查是否排除了合理候选或降低了风险"})
    for e in events:
        h = e["hindsight"] or {}
        if h.get("outcome") == "重复已有" and h.get("ex_ante_reason") != "有依据" and not h.get("valuable_if_failed"):
            run.append(e)
        else:
            emit()
            run = []
    emit()
    return found


def make_proposed_action(e, visible, content_chars):
    msg = visible[e["start"]]
    if e["kind"] == "tool_call":
        return {"kind": "tool_call", "role": e["role"], "tool_name": e["tool_name"],
                "arguments": e["arguments"][:content_chars], "arguments_truncated": len(e["arguments"]) > content_chars,
                "message_index": e["start"], "related_event_ids": e["related_event_ids"]}
    return {"kind": e["kind"], "role": e["role"], "message_index": e["start"],
            "content": msg.get("content", ""), "content_truncated": msg.get("content_truncated", False)}


def process(rec, args):
    messages = rec["messages"]
    if not messages:
        raise ValueError("空轨迹")
    events = build_events(messages)
    visible = [compact_message(m, i, args.content_chars) for i, m in enumerate(messages)]
    refs = references(rec["task_id"], not args.no_reference_cases)
    total = cost(messages, range(len(messages)))
    r = {"method_version": "2.0", "task_id": rec["task_id"], "message_count": len(messages),
         "event_count": len(events), "events": events, "total_cost": total, "reference_case_sources":
         [x["source"] for x in refs], "reference_case_policy": "只使用其他任务经原轨迹核对的人工案例；不读取当前任务人工标签；同任务样本自动排除。",
         "truncated_message_indices": [m["i"] for m in visible if m["content_truncated"]],
         "scan_missing_ids": [], "selected_pre_action_ids": [], "reviewed_pre_action_ids": [],
         "pending_candidate_ids": [], "unselected_event_ids": [], "manual_review_ids": [],
         "suspected_stagnation_stages": [], "backtrack_candidate_ids": [],
         "model_calls": [], "annotation_cost": {}, "status": "dry_run" if args.dry_run else "partial",
         "caveat": "区间由程序划分，标签是新的模型判断，不是人工金标。返回字符不是 token；模型耗时/usage 是标注成本，不是原轨迹执行成本。"}
    if args.dry_run:
        return r
    last = len(messages) - 1
    for offset in range(0, len(events), args.scan_batch_events):
        batch = events[offset:offset + args.scan_batch_events]
        payload = {"task_id": rec["task_id"], "reference_cases": refs, "messages": visible,
                   "target_events": [{k: e[k] for k in ("id", "kind", "role", "start", "end", "message_indices",
                                                         "tool_name", "return_indices", "rule_hints", "related_event_ids")
                                      if k in e} for e in batch]}
        prompt = SCAN + json.dumps(payload, ensure_ascii=False)
        if len(prompt) > args.max_prompt_chars:
            for e in batch:
                e["hindsight_error"] = f"初筛提示词 {len(prompt)} 字符超上限，未发送"
            continue
        obj, meta = call_model(prompt, max_tokens=args.max_completion_tokens)
        r["model_calls"].append({"stage": "full_scan", "event_ids": [e["id"] for e in batch], **meta})
        if obj is None:
            for e in batch:
                e["hindsight_error"] = meta.get("error", "模型初筛失败")
            continue
        by_id = {e["id"]: e for e in batch}
        for item in obj.get("events", []) if isinstance(obj.get("events"), list) else []:
            if not isinstance(item, dict) or item.get("id") not in by_id:
                continue
            e = by_id.pop(item["id"])
            normalize_judgment(item)
            error = validate_scan(item, e, last)
            if isinstance(item.get("backtrack_from_indices"), list):
                links = item["backtrack_from_indices"]
                if links and all(type(i) is int and 0 <= i < e["start"] for i in links):
                    e["backtrack_from_indices"] = links
                elif links:
                    item["needs_review"] = True
                    item.setdefault("evidence_warning", []).append("回退证据下标无效")
            if item.get("observed_failure") and not any(messages[i].get("error") for i in e["return_indices"]):
                item["needs_review"] = True
                item.setdefault("evidence_warning", []).append("原轨迹未显示工具报错，模型称失败需核对")
            if error:
                e["hindsight_error"] = error
            else:
                e["hindsight"] = item
        for e in by_id.values():
            e["hindsight_error"] = "初筛漏掉此事件或重复返回了同一 id"
    r["scan_missing_ids"] = [e["id"] for e in events if e["hindsight"] is None]
    r["suspected_stagnation_stages"] = scan_stage_candidates(events, messages)
    r["backtrack_candidate_ids"] = [e["id"] for e in events if e.get("backtrack_from_indices")]
    chosen, pending, controls = choose_targets(events, args.max_prefix, args.control_events)
    r["selected_pre_action_ids"] = [e["id"] for e in chosen]
    r["pending_candidate_ids"] = pending
    r["control_sample_ids"] = controls
    chosen_ids = {e["id"] for e in chosen}
    r["unselected_event_ids"] = [e["id"] for e in events if e["id"] not in chosen_ids]
    for e in chosen:
        # 工具动作绝不看到本次调用/返回；只给拟执行调用的名字和参数。
        # 用户/助手纯文字动作在 proposed_action 给出本条内容，但依旧不泄漏未来。
        payload = {"task_id": rec["task_id"], "reference_cases": refs,
                   "target": {"id": e["id"], "kind": e["kind"], "start": e["start"]},
                   "messages_before_action": visible[:e["start"]],
                   "proposed_action": make_proposed_action(e, visible, args.content_chars)}
        prompt = BEFORE + json.dumps(payload, ensure_ascii=False)
        if len(prompt) > args.max_prompt_chars:
            e["prefix_error"] = f"调用前提示词 {len(prompt)} 字符超上限，未发送"
            continue
        obj, meta = call_model(prompt, max_tokens=args.max_completion_tokens)
        r["model_calls"].append({"stage": "before_action", "event_id": e["id"], **meta})
        if obj is None:
            e["prefix_error"] = meta.get("error", "调用前复核失败")
            continue
        normalize_judgment(obj)
        # 当前拟动作不属于可引用历史；不强制要求 evidence_indices 包含目标事件。
        if obj.get("label") not in LABELS or not basic_evidence(obj, e["start"] - 1):
            e["prefix_error"] = "调用前标签或历史证据下标非法；检查 raw"
            continue
        if obj.get("ex_ante_reason") not in EX_ANTE:
            obj["ex_ante_reason"] = "不确定"
            obj["needs_review"] = True
        e["prefix"] = obj
        e["prefix_viewpoint"] = "before_action_no_current_result"
        r["reviewed_pre_action_ids"].append(e["id"])
    for e in events:
        h, before = e.get("hindsight"), e.get("prefix")
        if (e["id"] in pending or e.get("rule_hints") or e.get("hindsight_error") or e.get("prefix_error")
            or h and (h.get("needs_review") or h.get("label") in ("低效重复", "证据不足")
                      or h.get("ex_ante_reason") != "有依据" or h.get("used_later") == "否")
            or before and (before.get("needs_review") or h and before.get("label") != h.get("label"))):
            r["manual_review_ids"].append(e["id"])
    r["annotation_cost"] = {"model_requests": len(r["model_calls"]),
        "attempts": sum(len(c.get("attempts", [])) for c in r["model_calls"]),
        "429_attempts": sum('429' in a.get("error", "") for c in r["model_calls"] for a in c.get("attempts", [])),
        "elapsed_s_sum": round(sum(a.get("elapsed_s", 0) for c in r["model_calls"] for a in c.get("attempts", [])), 3),
        "usage_returned_attempts": sum(bool(a.get("usage")) for c in r["model_calls"] for a in c.get("attempts", [])),
        "usage_missing_attempts": sum(not a.get("usage") for c in r["model_calls"] for a in c.get("attempts", []))}
    r["status"] = ("partial" if r["scan_missing_ids"] or len(r["reviewed_pre_action_ids"]) != len(chosen)
                   else "screened_with_pending" if pending else "screened")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--task-id")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-reference-cases", action="store_true")
    ap.add_argument("--content-chars", type=int, default=3500, help="默认保留长航班结果；截断下标在输出中明确记录")
    ap.add_argument("--scan-batch-events", type=int, default=32, help="每次模型初筛的事件数；task 18 默认一次覆盖全部")
    ap.add_argument("--max-prefix", type=int, default=6, help="最多调用前深查几个优先事件；0=所有候选，不包括普通段")
    ap.add_argument("--control-events", type=int, default=1, help="从非候选工具事件抽样对照，默认 1")
    ap.add_argument("--max-prompt-chars", type=int, default=150000)
    ap.add_argument("--max-completion-tokens", type=int, default=6500)
    args = ap.parse_args()
    if args.limit < 0 or args.max_prefix < 0 or args.control_events < 0 or args.scan_batch_events < 1 or args.content_chars < 100:
        ap.error("limit/max-prefix/control-events 不得为负，scan-batch-events 至少 1，content-chars 至少 100")
    if not args.input.is_file():
        ap.error(f"找不到输入文件：{args.input}")
    if args.output.resolve() == args.input.resolve() or args.output.resolve() == Path(__file__).resolve():
        ap.error("不得覆盖输入或脚本")
    if args.output.exists() and not args.overwrite:
        ap.error("输出已有文件：换名字，或显式加 --overwrite")
    if not args.dry_run and not os.environ.get("GATEWAY_SK", "").strip():
        ap.error("缺少 GATEWAY_SK；密钥只放环境变量，不要写进脚本")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", encoding="utf-8") as f:
        os.chmod(args.output, 0o600)
        for rec in load_traces(args.input):
            if args.task_id and rec["task_id"] != args.task_id:
                continue
            if args.limit and n >= args.limit:
                break
            result = process(rec, args)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print(f"[{n}] task_id={rec['task_id']} events={result['event_count']} "
                  f"requests={len(result['model_calls'])} pre_action={len(result['reviewed_pre_action_ids'])} "
                  f"pending={len(result['pending_candidate_ids'])} status={result['status']}", file=sys.stderr)
    if not n:
        print("没有匹配的 task_id", file=sys.stderr)
        return 1
    print(f"完成：{args.output}。这是新版本，不覆盖旧版结果。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
