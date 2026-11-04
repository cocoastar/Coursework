#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轨迹冗余标注 v3：规则预筛 → 一次批量"决策影响"事后判断 → 选择性复核。

与 v1/v2 的根本区别：
  - 判据从"调用前是否有理由"改为"这步结果能否影响任何后续决策（含排除分支）"。
    v1/v2 的 prefix 复核屏蔽未来、只看当时理由，任何 search 都能编出理由，
    于是系统性把冗余漂白成"必要验证"。v3 事后判，并把后续写操作参数喂进去当锚点。
  - 口径对齐 benchmark：输出 Duplicated/Abnormal/Incorrect/Exploratory + 非冗余，
    删掉自造的"必要验证"（它是过宽的根源，benchmark 标签体系里也没有这一类）。
  - 成本：规则预筛(0 请求) + 一次批量判断(1 请求) + 选择性复核(0-1 请求)，
    目标 1-2 请求/轨迹，远低于 v1 的 ~31、v2 的 ~7。

不读取当前 task 的 annotation.json 作为答案；few-shot 参考案例来自其它 task 的人工标注，
按 task_id 自动排除同任务。密钥仅来自 GATEWAY_SK。
"""
import argparse
import json
import os
import sys
from pathlib import Path

from trace_method import (LABELS as _V1_LABELS, build_events, call_model, compact_message,
                          cost, load_traces, parse_json, text)

# benchmark 四类冗余 + 非冗余。中文标签与 benchmark 英文类型的映射见 LABEL_TO_BENCH。
REDUNDANT_LABELS = ("低效重复", "错误恢复", "无依据", "探索未用")
LABELS = REDUNDANT_LABELS + ("非冗余",)
LABEL_TO_BENCH = {
    "低效重复": "Duplicated Step",
    "错误恢复": "Abnormal Step",
    "无依据": "Incorrect Step",
    "探索未用": "Exploratory Step",
}

# 写操作关键词：这些工具会改变系统状态；其 arguments 是"决策影响"的锚点。
WRITE_PREFIXES = ("update_", "book_", "cancel_", "modify_", "make_payment", "return_",
                  "exchange_", "send_payment", "reset_", "toggle_", "enable_", "reboot_",
                  "resume_", "grant_", "reseat_", "set_", "refuel_", "disconnect_")
# 读/查类：结果是否被用上需要判别，是候选的主要来源。
READ_PREFIXES = ("get_", "search_", "check_", "find_", "list_", "run_speed", "can_send")

# 经原始轨迹核对的四类人工案例，仅解释判别边界；同 task_id 自动排除，绝不喂当前答案。
# 不含被评价轨迹的标签。lesson 均来自对真实轨迹的核对，不引用空/合成 reason。
REFERENCE_CASES = (
    {"source": "airline task 5 [6-7]", "bench_type": "Exploratory Step", "evidence": [12, 13],
     "lesson": "目标 HAT045 未定位时查候选预订 DB1Y70 当时合理；但它不含 HAT045，"
               "最终只用了 [12-13] 命中的 3JA7XV。删掉它不改变任何后续决策——它没排除掉"
               "'下一步该查哪张'，只是一次未命中。判探索未用(Exploratory)：当时有理由，结果未影响决策。"
               "evidence 填 [12,13]（真正被用到的那组），不是本步自己。"},
    {"source": "airline task 0 [4-5]", "bench_type": "Incorrect Step", "evidence": [2, 3],
     "lesson": "目标预订 EHGLP3 已在 [2-3] 定位且字段齐全，仍去查会员/账户资料；这些对取消分支"
               "没有贡献，也非合理候选搜索。判无依据(Incorrect)：目标已定后无理由的查询。"
               "evidence 填 [2,3]（证明目标与所需信息已齐），不是本步自己。"},
    {"source": "airline task 14 [12-13]", "bench_type": "Duplicated Step", "evidence": [10, 11],
     "lesson": "与 [10-11] 同工具、同参数、同返回(HAT023)，零新增信息，期间无状态变化。"
               "判低效重复(Duplicated)：之前相同调用已有有效结果。evidence 填 [10,11]（更早那次相同调用）。"
               "注意同工具不同参数不算重复。"},
    {"source": "airline task 5 [8-9]", "bench_type": "Abnormal Step", "evidence": [10, 11],
     "lesson": "[8] 查 MUGYUB 遇 Network error，[10] 原样重试成功。失败调用触发了正确恢复，"
               "判错误恢复(Abnormal)：不归罪助手。evidence 填 [10,11]（成功重试那步）。"
               "必须真有报错才可判此类，没看到报错不能编造。"},
)

JUDGE = """你在给完整 agent 轨迹做事后冗余标注。轨迹已按每次工具调用及其返回拆成细事件。
核心判据只有一个：**这一步(工具调用+返回)的结果，能不能影响任何后续决策？**
- 若能指出它影响了某个具体决策——喂给了某条写操作的参数(给下标)、改变了下一步查哪个对象/排除了一个候选(给下标)、
  喂给了一个真实的分支选择或用户可见的结论(给下标)——则判「非冗余」。
- 注意："做题排除一个错误选项"是有影响的决策(它改变了下一步)；但若目标/对象/参数早已定死、
  这一步的结果无论是什么都不会改变后面实际发生的任何动作，那它就没有影响决策。
- 若指不出任何影响，再按以下区分冗余类型：
  * 当时有合理探索理由(目标未定、是个合理候选)但结果未影响决策 → 「探索未用」
  * 目标已定、无合理理由仍查无关对象 → 「无依据」
  * 之前相同工具+相同参数已有有效结果、状态未变、再做无新收益 → 「低效重复」
  * 工具真实报错后的重试链(失败步本身) → 「错误恢复」(必须真有报错，没看到不能编造)
我会把后续真实发生的写操作参数放在 write_actions 里当锚点：对照"搜出来的东西"是否真的进了写操作。
但对照是语义的，不是字符串匹配：原始对象本来就会出现在写操作里，不代表这次查询影响了决策。
不要因为工具名/参数不同就判冗余；相同也要考虑状态变化、失败重试、新用户要求。不要用"最后没成功"倒推无效。
reference_cases 是其它任务的人工判别示范，不是本任务答案。
evidence_indices 是**支撑你这条判断的那组下标**，不是被判步骤本身——绝不要把 target 自己的下标填进去。
  * 判冗余(探索未用/低效重复/无依据)：填那条证明"删掉本步也不影响处理"的下标。
    常见是对应的写操作组(如 update/cancel 的[调用,返回]，证明它用原参数、没用本步结果)，
    或任务目标约束所在的用户消息下标(证明本步偏离目标)，或更早那条相同调用(证明重复)。
  * 判错误恢复：填成功重试那一步的下标。
  * 判非冗余：填本步结果真正影响到的那条决策/写操作下标。
只输出 JSON：{"events":[{"id":"E...","label":"低效重复/错误恢复/无依据/探索未用/非冗余",
"decision_impact":"指出被哪条下标的决策使用；非冗余必填具体下标，冗余填空字符串",
"evidence_pair":[最关键的一组证据下标：工具调用组填[调用,返回]两个相邻下标；若证据是单条用户/目标消息则填[该下标]],
"evidence_indices":[支撑判断的全部下标，不含 target 自身],"observed_failure":false,
"reason":"一句话，用决策影响的语言，不得夸大必要性","confidence":0.0到1.0,"needs_review":false}]}
evidence_pair 是 evidence_indices 里最能支撑判断的那一组（评分只认这一组），务必精确：
参考案例的 evidence 字段就是正确的 evidence_pair 示范（如 [12,13]、[2,3]、[10,11]）。
每个 target 的 id 恰好出现一次。输入：\n"""


def references(task_id):
    return [{"source": c["source"], "type": c["bench_type"], "evidence": c["evidence"],
             "lesson": c["lesson"]}
            for c in REFERENCE_CASES if f"task {task_id} " not in c["source"]]


def is_write(name):
    n = (name or "").lower()
    return any(n.startswith(p) for p in WRITE_PREFIXES)


def is_read(name):
    n = (name or "").lower()
    return any(n.startswith(p) for p in READ_PREFIXES)


def prefilter(events, messages):
    """阶段 A（0 请求）：规则把事件分成送判候选 / 低嫌疑旁证。
    候选 = 读/查类工具调用 + 带 rule_hints(重复/报错/未配对) 的调用；
    写操作与纯文本默认非候选，但写操作参数要留作锚点。"""
    candidates, write_actions = [], []
    for e in events:
        if e["kind"] == "tool_call" and is_write(e.get("tool_name", "")):
            write_actions.append({"id": e["id"], "index": e["start"], "tool_name": e["tool_name"],
                                  "arguments": e.get("arguments", "")})
        reasons = []
        if e["kind"] == "tool_call" and is_read(e.get("tool_name", "")):
            reasons.append("读/查类调用：结果是否影响决策需判别")
        if e.get("rule_hints"):
            reasons.append("规则提示：" + "；".join(e["rule_hints"]))
        if e["kind"] == "unpaired_tool_return":
            reasons.append("工具返回未配对")
        if reasons:
            e["prefilter_reason"] = "；".join(reasons)
            candidates.append(e)
    return candidates, write_actions


def judge_payload(rec, batch, visible, write_actions, content_chars):
    return {
        "task_id": rec["task_id"],
        "reference_cases": references(rec["task_id"]),
        "messages": visible,
        "write_actions": write_actions,
        "target_events": [
            {k: e[k] for k in ("id", "kind", "role", "start", "end", "message_indices",
                               "tool_name", "return_indices", "rule_hints", "related_event_ids",
                               "prefilter_reason") if k in e}
            | ({"arguments": e["arguments"][:content_chars],
                "arguments_truncated": len(e["arguments"]) > content_chars} if "arguments" in e else {})
            for e in batch]}


def valid_indices(item, max_index):
    idx = item.get("evidence_indices")
    return isinstance(idx, list) and all(type(i) is int and 0 <= i <= max_index for i in idx)


def validate(item, event, last_index):
    if item.get("label") not in LABELS or not valid_indices(item, last_index):
        return "标签或证据下标非法"
    warnings = []
    c = item.get("confidence")
    if not isinstance(c, (int, float)) or not 0 <= c <= 1:
        item["confidence"] = 0.5
        warnings.append("confidence 缺失/越界，置 0.5")
    if item["label"] == "低效重复" and not any(i < event["start"] for i in item["evidence_indices"]):
        warnings.append("低效重复缺少较早的证据下标")
    # evidence 不应指向被判步骤自身（benchmark 口径：evidence 是支撑依据，非 target 本身）。
    if item["label"] in REDUNDANT_LABELS and set(item["evidence_indices"]) <= set(event["message_indices"]):
        warnings.append("evidence 只指向了本步自身，应指向支撑判断的其它下标")
    # 注："错误恢复"必须真有报错，用真实 messages 的核对放在 process() 里。
    if item["label"] == "非冗余" and not str(item.get("decision_impact") or "").strip():
        warnings.append("判非冗余但未给出决策影响下标")
    if warnings:
        item["needs_review"] = True
        item.setdefault("warnings", []).extend(warnings)
    return None


def confirm_payload(rec, event, visible, write_actions, content_chars):
    """阶段 C：仅对'判非冗余但规则强烈存疑'的事件复核一次，强制其证明决策影响。"""
    return {"task_id": rec["task_id"], "reference_cases": references(rec["task_id"]),
            "messages": visible, "write_actions": write_actions,
            "target_events": [{k: event[k] for k in ("id", "kind", "role", "start", "end",
                              "message_indices", "tool_name", "return_indices", "rule_hints",
                              "prefilter_reason") if k in event}
                              | ({"arguments": event["arguments"][:content_chars]} if "arguments" in event else {})]}


def process(rec, args):
    messages = rec["messages"]
    if not messages:
        raise ValueError("空轨迹")
    events = build_events(messages)
    visible = [compact_message(m, i, args.content_chars) for i, m in enumerate(messages)]
    candidates, write_actions = prefilter(events, messages)
    refs = references(rec["task_id"])
    total = cost(messages, range(len(messages)))
    last = len(messages) - 1

    r = {"method_version": "3.0", "task_id": rec["task_id"], "message_count": len(messages),
         "event_count": len(events), "events": events, "total_cost": total,
         "candidate_ids": [e["id"] for e in candidates],
         "write_action_ids": [w["id"] for w in write_actions],
         "reference_case_sources": [x["source"] for x in refs],
         "reference_case_policy": "仅用其它任务经轨迹核对的人工案例；按 task_id 排除同任务；不读当前任务人工标签。",
         "truncated_message_indices": [m["i"] for m in visible if m["content_truncated"]],
         "judged_ids": [], "confirmed_ids": [], "missing_ids": [], "manual_review_ids": [],
         "model_calls": [], "annotation_cost": {},
         "status": "dry_run" if args.dry_run else "partial",
         "caveat": "标签是新的模型判断，不是人工金标。返回字符不是 token；模型 usage/耗时是标注成本，不是原轨迹执行成本。"}
    if args.dry_run:
        return r

    # 阶段 B：一次（或按 batch 多次）把候选送"决策影响"判断
    for offset in range(0, len(candidates), args.scan_batch_events):
        batch = candidates[offset:offset + args.scan_batch_events]
        payload = judge_payload(rec, batch, visible, write_actions, args.content_chars)
        prompt = JUDGE + json.dumps(payload, ensure_ascii=False)
        if len(prompt) > args.max_prompt_chars:
            for e in batch:
                e["judge_error"] = f"判断提示词 {len(prompt)} 字符超上限，未发送"
            continue
        obj, meta = call_model(prompt, max_tokens=args.max_completion_tokens)
        r["model_calls"].append({"stage": "judge", "event_ids": [e["id"] for e in batch], **meta})
        if obj is None:
            for e in batch:
                e["judge_error"] = meta.get("error", "模型判断失败")
            continue
        by_id = {e["id"]: e for e in batch}
        for item in obj.get("events", []) if isinstance(obj.get("events"), list) else []:
            if not isinstance(item, dict) or item.get("id") not in by_id:
                continue
            e = by_id.pop(item["id"])
            error = validate(item, e, last)
            # 用真实 messages 核对"错误恢复"必须真有报错
            if item.get("label") == "错误恢复" and not any(
                    messages[i].get("error") for i in e.get("return_indices", []) if i <= last):
                item["needs_review"] = True
                item.setdefault("warnings", []).append("判错误恢复但原轨迹该步未见报错，需核对")
            if error:
                e["judge_error"] = error
            else:
                e["judgment"] = item
                r["judged_ids"].append(e["id"])
        for e in by_id.values():
            e["judge_error"] = "初判漏掉此事件或重复返回了同一 id"
    r["missing_ids"] = [e["id"] for e in candidates if e.get("judgment") is None and not e.get("judge_error")]

    # 阶段 C：对"判非冗余但规则存疑（重复/报错提示）"的少数强制复核
    suspicious = [e for e in candidates
                  if (e.get("judgment") or {}).get("label") == "非冗余"
                  and e.get("rule_hints") and len(r["confirmed_ids"]) < args.max_confirm]
    for e in suspicious[:args.max_confirm]:
        payload = confirm_payload(rec, e, visible, write_actions, args.content_chars)
        prompt = (JUDGE + "注意：此事件规则上存疑(重复/报错)，若判非冗余必须给出确切的决策影响下标。\n"
                  + json.dumps(payload, ensure_ascii=False))
        if len(prompt) > args.max_prompt_chars:
            e["confirm_error"] = "复核提示词超上限"
            continue
        obj, meta = call_model(prompt, max_tokens=args.max_completion_tokens)
        r["model_calls"].append({"stage": "confirm", "event_id": e["id"], **meta})
        if obj is None:
            e["confirm_error"] = meta.get("error", "复核失败")
            continue
        items = obj.get("events") if isinstance(obj.get("events"), list) else []
        item = next((x for x in items if isinstance(x, dict) and x.get("id") == e["id"]), None)
        if item and validate(item, e, last) is None:
            e["judgment_confirm"] = item
            if item.get("label") != e["judgment"]["label"]:
                e["judgment"]["needs_review"] = True
                e["judgment"].setdefault("warnings", []).append(
                    f"复核改判为 {item.get('label')}，原判 {e['judgment']['label']}")
            r["confirmed_ids"].append(e["id"])

    r["manual_review_ids"] = [e["id"] for e in events if e.get("judge_error") or e.get("confirm_error")
                              or (e.get("judgment") or {}).get("needs_review")]
    attempts = [a for c in r["model_calls"] for a in c.get("attempts", [])]
    r["annotation_cost"] = {
        "model_requests": len(r["model_calls"]),
        "attempts": len(attempts),
        "429_attempts": sum("429" in a.get("error", "") for a in attempts),
        "elapsed_s_sum": round(sum(a.get("elapsed_s", 0) for a in attempts), 3),
        "usage_returned_attempts": sum(bool(a.get("usage")) for a in attempts)}
    r["status"] = "ok" if r["judged_ids"] and not r["missing_ids"] else "partial"
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--task-id")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--content-chars", type=int, default=2000)
    ap.add_argument("--scan-batch-events", type=int, default=64, help="每次判断的候选事件数；默认一次覆盖全部")
    ap.add_argument("--max-confirm", type=int, default=3, help="阶段C最多复核几个存疑的非冗余判定")
    ap.add_argument("--max-prompt-chars", type=int, default=150000)
    ap.add_argument("--max-completion-tokens", type=int, default=6500)
    args = ap.parse_args()
    if args.limit < 0 or args.max_confirm < 0 or args.scan_batch_events < 1 or args.content_chars < 100:
        ap.error("limit/max-confirm 不得为负，scan-batch-events 至少 1，content-chars 至少 100")
    if not args.input.is_file():
        ap.error(f"找不到输入文件：{args.input}")
    if args.output.resolve() == args.input.resolve() or args.output.resolve() == Path(__file__).resolve():
        ap.error("不得覆盖输入或脚本")
    if args.output.exists() and not args.overwrite:
        ap.error("输出已有文件：换名字或加 --overwrite")
    if not args.dry_run and not os.environ.get("GATEWAY_SK", "").strip():
        ap.error("缺少 GATEWAY_SK；密钥只放环境变量")
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
                  f"candidates={len(result['candidate_ids'])} requests={len(result['model_calls'])} "
                  f"judged={len(result['judged_ids'])} status={result['status']}", file=sys.stderr)
    if not n:
        print("没有匹配的 task_id", file=sys.stderr)
        return 1
    print(f"完成：{args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
