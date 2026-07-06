#!/usr/bin/env python3
"""Export OpenAI-format trajectories from a trace.db.

One session → one trajectory. We take each session's main-agent thread,
flatten it into a standard OpenAI chat-completions conversation (system /
user / assistant / tool). All steps are kept verbatim — no cleaning is
applied.

Outputs (per session_id) under <out_dir>/:
    <session>.json    flat trajectory, usable as an SFT sample ({messages, tools}).

The DB is either the OpenAI/litellm trace (older runs) or a passthrough
Anthropic trace (66/trace.db) — we always work from the Anthropic source
(request_body_json + response_body_json) so both shapes are supported.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


KEPT_ROLES = ("main",)

OPSNAME_PATTERNS = [
    re.compile(r"算子描述文件为\s*\S*?/(\d+_[A-Za-z][A-Za-z0-9_]*)\.py"),
    re.compile(r"输出到\s*\S*?/(\d+_[A-Za-z][A-Za-z0-9_]*)/?"),
    re.compile(r"/level\d+/(\d+_[A-Za-z][A-Za-z0-9_]*)\.py"),
]
_FNAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def extract_opsname(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Scan the first few user messages for the operator-name pattern used in
    the Ascend op-gen prompts. Returns sanitized opsname or None."""
    blob = []
    for m in messages[:8]:
        if m.get("role") == "user":
            content = m.get("content") or ""
            if isinstance(content, str):
                blob.append(content)
        if sum(len(s) for s in blob) > 8000:
            break
    text = "\n".join(blob)
    for pat in OPSNAME_PATTERNS:
        m = pat.search(text)
        if m:
            name = m.group(1)
            return _FNAME_SAFE_RE.sub("_", name).strip("._-") or None
    return None



# ---------------------------------------------------------------------------
# Anthropic → OpenAI conversion
# ---------------------------------------------------------------------------

def _system_to_text(sys_field: Any) -> str:
    if sys_field is None:
        return ""
    if isinstance(sys_field, str):
        return sys_field
    if isinstance(sys_field, list):
        parts = []
        for blk in sys_field:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                parts.append(blk["text"])
            elif isinstance(blk, str):
                parts.append(blk)
        return "\n".join(parts)
    return str(sys_field)


def _content_to_text(content: Any) -> str:
    """Flatten Anthropic content (str or list of blocks) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                t = blk.get("type")
                if t == "text" and isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
                elif t == "image":
                    parts.append("[image]")
                elif t == "thinking" and isinstance(blk.get("thinking"), str):
                    parts.append(blk["thinking"])
                else:
                    # tool_use / tool_result handled separately; ignore here
                    pass
        return "\n".join(p for p in parts if p)
    return str(content)


def _tool_result_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for blk in content:
            if isinstance(blk, dict):
                t = blk.get("type")
                if t == "text" and isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
                elif t == "image":
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(blk, ensure_ascii=False))
            else:
                parts.append(str(blk))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _tool_use_to_openai_call(blk: Dict[str, Any]) -> Dict[str, Any]:
    raw_input = blk.get("input")
    try:
        arguments = json.dumps(raw_input or {}, ensure_ascii=False)
    except (TypeError, ValueError):
        arguments = json.dumps({"_raw": str(raw_input)}, ensure_ascii=False)
    return {
        "id": blk.get("id"),
        "type": "function",
        "function": {
            "name": blk.get("name"),
            "arguments": arguments,
        },
    }


def anthropic_message_to_openai(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert one Anthropic message into 0+ OpenAI-format messages.

    User messages containing tool_result blocks expand into one `role:"tool"`
    message per tool_result, plus an optional user message for any text.
    Assistant messages with tool_use blocks become a single assistant message
    with `tool_calls`, with text + thinking concatenated into `content`.
    """
    role = msg.get("role")
    content = msg.get("content")

    # User message: split into tool messages + optional user text.
    if role == "user":
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        if not isinstance(content, list):
            return [{"role": "user", "content": _content_to_text(content)}]

        out: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        for blk in content:
            if not isinstance(blk, dict):
                if isinstance(blk, str):
                    text_parts.append(blk)
                continue
            t = blk.get("type")
            if t == "tool_result":
                text = _tool_result_content_to_text(blk.get("content"))
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": blk.get("tool_use_id"),
                    "content": text,
                }
                if blk.get("is_error"):
                    tool_msg["_is_error"] = True  # stripped before final emit
                out.append(tool_msg)
            elif t == "text" and isinstance(blk.get("text"), str):
                text_parts.append(blk["text"])
            elif t == "image":
                text_parts.append("[image]")
        text_joined = "\n".join(p for p in text_parts if p).strip()
        if text_joined:
            out.append({"role": "user", "content": text_joined})
        return out

    # Assistant message: collapse thinking+text into content, tool_use → tool_calls.
    if role == "assistant":
        thinking_parts: List[str] = []
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict):
                    if isinstance(blk, str):
                        text_parts.append(blk)
                    continue
                t = blk.get("type")
                if t == "text" and isinstance(blk.get("text"), str):
                    text_parts.append(blk["text"])
                elif t == "thinking" and isinstance(blk.get("thinking"), str):
                    thinking_parts.append(blk["thinking"])
                elif t == "tool_use":
                    tool_calls.append(_tool_use_to_openai_call(blk))

        parts: List[str] = []
        if thinking_parts:
            parts.append("<think>\n" + "\n".join(thinking_parts).strip() + "\n</think>")
        if text_parts:
            parts.append("\n".join(text_parts).strip())
        content_str = "\n\n".join(p for p in parts if p) or None

        out_msg: Dict[str, Any] = {"role": "assistant", "content": content_str}
        if tool_calls:
            out_msg["tool_calls"] = tool_calls
        return [out_msg]

    # Anything else: best-effort
    return [{"role": role or "user", "content": _content_to_text(content)}]


def anthropic_response_to_openai_message(response: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(response, dict):
        return None
    msg = {"role": "assistant", "content": response.get("content")}
    converted = anthropic_message_to_openai(msg)
    return converted[0] if converted else None


def anthropic_tools_to_openai(tools: Any) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(tools, list):
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description"),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# Build the raw flat trajectory for a session
# ---------------------------------------------------------------------------

def _get_anthropic_body(row: sqlite3.Row) -> Dict[str, Any]:
    """Return the original Anthropic request body regardless of trace mode."""
    if row["request_body_json"]:
        try:
            return json.loads(row["request_body_json"])
        except json.JSONDecodeError:
            pass
    if row["converted_request_json"]:
        try:
            conv = json.loads(row["converted_request_json"])
        except json.JSONDecodeError:
            conv = {}
        if isinstance(conv, dict) and isinstance(conv.get("body"), dict):
            return conv["body"]
    return {}


COMPACTION_MARKER = "[--- context compaction boundary (Claude Code /compact) ---]"


def _segment_main_requests(rows: List[sqlite3.Row]) -> List[List[int]]:
    """Split chronologically-ordered main requests into segments at compaction
    points. A compaction is signaled by message_count dropping vs the prior
    request. Within a segment, message_count grows monotonically; we keep only
    the last request of each segment (the tip of that segment's chain)."""
    segments: List[List[int]] = []
    current: List[int] = []
    prev_count = -1
    for i, r in enumerate(rows):
        mc = r["message_count"] or 0
        if current and mc < prev_count:
            segments.append(current)
            current = []
        current.append(i)
        prev_count = mc
    if current:
        segments.append(current)
    return segments


_TRAJECTORY_SELECT = """
    SELECT trace_id, message_count, started_ms,
           request_body_json, converted_request_json, response_body_json,
           model_requested, model_mapped, response_stop_reason
"""


def _build_trajectory_from_rows(
    rows: List[sqlite3.Row],
    session_row: sqlite3.Row,
    kind_label: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Assemble one flat OpenAI trajectory from a chronologically ordered list
    of request rows. Detects compaction (message_count drops) and stitches
    each segment together with a boundary marker. Returns None if there is
    no usable content."""
    if not rows:
        return None

    segments = _segment_main_requests(rows)
    segment_tips = [rows[seg[-1]] for seg in segments]

    messages: List[Dict[str, Any]] = []
    tools: List[Dict[str, Any]] = []
    segment_summaries: List[Dict[str, Any]] = []

    for seg_idx, (seg_indices, tip) in enumerate(zip(segments, segment_tips)):
        body = _get_anthropic_body(tip)
        if not body:
            continue

        seg_message_start = len(messages)

        if seg_idx == 0:
            system_text = _system_to_text(body.get("system"))
            if system_text:
                messages.append({"role": "system", "content": system_text})
        else:
            messages.append({
                "role": "system",
                "content": COMPACTION_MARKER,
                "_compaction_marker": True,
            })

        for am in body.get("messages") or []:
            if isinstance(am, dict):
                messages.extend(anthropic_message_to_openai(am))

        response_body = None
        if tip["response_body_json"]:
            try:
                response_body = json.loads(tip["response_body_json"])
            except json.JSONDecodeError:
                response_body = None
        final_assistant = anthropic_response_to_openai_message(response_body)
        if final_assistant:
            messages.append(final_assistant)

        tools = anthropic_tools_to_openai(body.get("tools")) or tools

        segment_summaries.append({
            "segment_index": seg_idx,
            "tip_trace_id": tip["trace_id"],
            "tip_message_count": tip["message_count"],
            "tip_stop_reason": tip["response_stop_reason"],
            "n_requests": len(seg_indices),
            "first_started_ms": rows[seg_indices[0]]["started_ms"],
            "last_started_ms": rows[seg_indices[-1]]["started_ms"],
            "openai_message_range": [seg_message_start, len(messages)],
        })

    meta: Dict[str, Any] = {
        "trajectory_kind": kind_label,
        "n_segments": len(segments),
        "n_compactions": max(0, len(segments) - 1),
        "n_requests": len(rows),
        "tip_trace_ids": [t["trace_id"] for t in segment_tips],
        "segments": segment_summaries,
        "model_requested": segment_tips[-1]["model_requested"] if segment_tips else None,
        "model_mapped": segment_tips[-1]["model_mapped"] if segment_tips else None,
        "final_stop_reason": segment_tips[-1]["response_stop_reason"] if segment_tips else None,
        "first_seen": session_row["first_seen"],
        "last_seen": session_row["last_seen"],
    }
    if extra_meta:
        meta.update(extra_meta)

    return {
        "session_id": session_row["session_id"],
        "tools": tools,
        "messages": messages,
        "_meta": meta,
    }


def build_raw_trajectory(conn: sqlite3.Connection, session_row: sqlite3.Row) -> Optional[Dict[str, Any]]:
    rows = conn.execute(
        f"""
        {_TRAJECTORY_SELECT}
        FROM requests
        WHERE session_id = ? AND role_kind = 'main' AND api = 'messages'
        ORDER BY started_ms ASC
        """,
        (session_row["session_id"],),
    ).fetchall()
    return _build_trajectory_from_rows(list(rows), session_row, "main")


def build_subagent_trajectories(
    conn: sqlite3.Connection, session_row: sqlite3.Row
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return [(agent_call_id, subagent_type, bundle), ...] for every Agent/Task
    invocation in this session. Each bundle is built from the subagent's own
    request stream, grouped by parent_agent_call_id."""
    sid = session_row["session_id"]
    acs = conn.execute(
        """
        SELECT agent_call_id, agent_label, tool_name, started_ms, status, input_preview
        FROM agent_calls
        WHERE session_id = ? AND tool_name IN ('Agent', 'Task')
        ORDER BY started_ms ASC
        """,
        (sid,),
    ).fetchall()
    if not acs:
        return []

    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for ac in acs:
        rows = conn.execute(
            f"""
            {_TRAJECTORY_SELECT}
            FROM requests
            WHERE session_id = ? AND role_kind = 'subagent'
              AND parent_agent_call_id = ? AND api = 'messages'
            ORDER BY started_ms ASC
            """,
            (sid, ac["agent_call_id"]),
        ).fetchall()
        if not rows:
            continue
        bundle = _build_trajectory_from_rows(
            list(rows),
            session_row,
            "subagent",
            extra_meta={
                "agent_call_id": ac["agent_call_id"],
                "subagent_type": ac["agent_label"],
                "agent_call_status": ac["status"],
                "agent_call_input_preview": ac["input_preview"],
            },
        )
        if bundle and len(bundle["messages"]) >= 2:
            out.append((ac["agent_call_id"], ac["agent_label"] or "subagent", bundle))
    return out



# ---------------------------------------------------------------------------
# JSON-input adapter: load one or more exported-DB JSON dumps into an
# in-memory SQLite so the rest of the pipeline runs unchanged.
# ---------------------------------------------------------------------------

_MEM_SCHEMA = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    origin TEXT, first_seen TEXT, last_seen TEXT,
    request_count INTEGER, cc_version TEXT, cch TEXT
);
CREATE TABLE requests (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT, api TEXT, role_kind TEXT, agent_label TEXT,
    model_requested TEXT, model_mapped TEXT,
    started_at TEXT, started_ms INTEGER,
    completed_at TEXT, completed_ms INTEGER, duration_ms INTEGER,
    status TEXT, status_code INTEGER,
    message_count INTEGER,
    prefix_hashes_json TEXT,
    request_body_json TEXT, converted_request_json TEXT,
    response_body_json TEXT, error_json TEXT,
    response_stop_reason TEXT,
    parent_trace_id TEXT, parent_agent_call_id TEXT,
    title TEXT
);
CREATE INDEX idx_req_session ON requests(session_id, started_ms);
CREATE TABLE agent_calls (
    agent_call_id TEXT PRIMARY KEY,
    session_id TEXT, parent_trace_id TEXT,
    tool_use_id TEXT, tool_name TEXT, agent_label TEXT,
    started_at TEXT, started_ms INTEGER,
    completed_at TEXT, completed_ms INTEGER, duration_ms INTEGER,
    status TEXT,
    input_preview TEXT, result_preview TEXT, result_trace_id TEXT
);
CREATE INDEX idx_ac_session ON agent_calls(session_id, started_ms);
"""


def _to_json_str(v: Any) -> Optional[str]:
    """Re-serialize dict/list back to JSON string for storage; pass through strings."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps(str(v), ensure_ascii=False)


_SESSION_COLS = ("session_id", "origin", "first_seen", "last_seen",
                 "request_count", "cc_version", "cch")
_REQ_COLS = (
    "trace_id", "session_id", "api", "role_kind", "agent_label",
    "model_requested", "model_mapped",
    "started_at", "started_ms",
    "completed_at", "completed_ms", "duration_ms",
    "status", "status_code", "message_count",
    "prefix_hashes_json", "request_body_json", "converted_request_json",
    "response_body_json", "error_json",
    "response_stop_reason",
    "parent_trace_id", "parent_agent_call_id", "title",
)
_AC_COLS = (
    "agent_call_id", "session_id", "parent_trace_id",
    "tool_use_id", "tool_name", "agent_label",
    "started_at", "started_ms", "completed_at", "completed_ms", "duration_ms",
    "status", "input_preview", "result_preview", "result_trace_id",
)
_JSON_FIELDS_IN_REQ = {
    "prefix_hashes_json", "request_body_json", "converted_request_json",
    "response_body_json", "error_json",
}


def _ingest_json_payload(conn: sqlite3.Connection, payload: Any, src: str) -> int:
    """Insert one exported JSON dump into the in-memory db. Returns
    the number of sessions ingested from this payload."""
    # Accept either {session, requests, agent_calls, ...} or a list of such.
    if isinstance(payload, list):
        return sum(_ingest_json_payload(conn, item, src) for item in payload)
    if not isinstance(payload, dict):
        return 0

    sess = payload.get("session") or {}
    if not isinstance(sess, dict) or not sess.get("session_id"):
        print(f"  warning: {src} has no session_id, skipping", file=sys.stderr)
        return 0

    sess_row = tuple(sess.get(c) for c in _SESSION_COLS)
    conn.execute(
        f"INSERT OR REPLACE INTO sessions ({','.join(_SESSION_COLS)}) "
        f"VALUES ({','.join('?' * len(_SESSION_COLS))})",
        sess_row,
    )

    for r in payload.get("requests") or []:
        if not isinstance(r, dict):
            continue
        row = []
        for c in _REQ_COLS:
            v = r.get(c)
            if c in _JSON_FIELDS_IN_REQ:
                v = _to_json_str(v)
            row.append(v)
        conn.execute(
            f"INSERT OR REPLACE INTO requests ({','.join(_REQ_COLS)}) "
            f"VALUES ({','.join('?' * len(_REQ_COLS))})",
            tuple(row),
        )

    for ac in payload.get("agent_calls") or []:
        if not isinstance(ac, dict):
            continue
        # Make sure session_id is set even if the dump omitted it on the row
        ac_local = dict(ac)
        ac_local.setdefault("session_id", sess.get("session_id"))
        row = tuple(ac_local.get(c) for c in _AC_COLS)
        conn.execute(
            f"INSERT OR REPLACE INTO agent_calls ({','.join(_AC_COLS)}) "
            f"VALUES ({','.join('?' * len(_AC_COLS))})",
            row,
        )
    return 1


def load_json_input(input_path: Path) -> sqlite3.Connection:
    """Build a temporary in-memory SQLite db from one JSON file or a directory
    of JSON files (one exported session per file)."""
    if input_path.is_dir():
        paths = sorted(input_path.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"no *.json files found in {input_path}")
    else:
        paths = [input_path]

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_MEM_SCHEMA)

    total = 0
    for p in paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  warning: skipping {p} ({e})", file=sys.stderr)
            continue
        total += _ingest_json_payload(conn, payload, str(p))
    conn.commit()
    print(f"loaded {total} session(s) from {input_path} into in-memory db")
    return conn



def _strip_internal_fields(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        clean_m = {k: v for k, v in m.items() if not k.startswith("_")}
        out.append(clean_m)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="cc_traces/trace.db",
                        help="Input: a trace.db SQLite file, OR an exported JSON dump "
                             "({session, requests, agent_calls, ...}), OR a directory of "
                             "such JSON files (one per session). Auto-detected by suffix.")
    parser.add_argument("--out", default="exports/training",
                        help="Output directory (default: exports/training)")
    parser.add_argument("--session", action="append",
                        help="Only export the given session_id(s).")
    parser.add_argument("--include-archived", action="store_true",
                        help="Also export archived sessions.")
    parser.add_argument("--name-by", choices=("opsname", "session_id"), default="opsname",
                        help="Filename basis. 'opsname' extracts the Ascend operator name "
                             "from the task brief and falls back to session_id when not found "
                             "(default). 'session_id' uses the raw session id.")
    parser.add_argument("--export-subagents", action="store_true",
                        help="Also export each Agent/Task subagent run as its own training "
                             "sample <base>__sub<i>_<subagent_type>.json.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"error: trace db not found: {db_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_json_input = (
        db_path.is_dir() or
        db_path.suffix.lower() in (".json", ".jsonl")
    )
    if is_json_input:
        conn = load_json_input(db_path)
    else:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

    session_cols = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
    has_archived = "archived" in session_cols

    where = []
    params: List[Any] = []
    if has_archived and not args.include_archived:
        where.append("archived = 0")
    if args.session:
        where.append(f"session_id IN ({','.join('?' * len(args.session))})")
        params.extend(args.session)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    session_rows = conn.execute(
        f"SELECT * FROM sessions {where_sql} ORDER BY last_seen DESC",
        params,
    ).fetchall()

    summary = {"total_sessions": 0, "written": 0, "skipped_no_main": 0,
               "total_msgs": 0,
               "named_by_opsname": 0, "named_by_session_fallback": 0,
               "sessions_with_compaction": 0, "total_compactions": 0,
               "subagents_written": 0, "subagent_msgs": 0}

    used_names: Dict[str, int] = {}

    for s in session_rows:
        summary["total_sessions"] += 1
        bundle = build_raw_trajectory(conn, s)
        if not bundle or len(bundle["messages"]) < 2:
            summary["skipped_no_main"] += 1
            continue

        raw_messages = bundle["messages"]
        session_id = bundle["session_id"]
        short_sid = session_id.split("-")[0] if "-" in session_id else session_id[:8]

        opsname = extract_opsname(raw_messages) if args.name_by == "opsname" else None
        if opsname:
            base = opsname
            bundle["_meta"]["opsname"] = opsname
            summary["named_by_opsname"] += 1
        else:
            base = session_id.replace("/", "_")
            bundle["_meta"]["opsname"] = None
            if args.name_by == "opsname":
                summary["named_by_session_fallback"] += 1
        # disambiguate collisions (e.g. two sessions for the same op)
        count = used_names.get(base, 0)
        if count == 0:
            safe_name = base
        else:
            safe_name = f"{base}__{short_sid}"
        used_names[base] = count + 1

        main_safe_name = safe_name  # used to derive subagent filenames

        def _emit(bundle_, safe_, indent="  ", extra_tail=""):
            """Write the trajectory for one bundle, return (n_msgs, n_comp)."""
            raw_msgs = bundle_["messages"]
            n_comp_ = bundle_["_meta"]["n_compactions"]
            n_seg_ = bundle_["_meta"]["n_segments"]
            n_req_ = bundle_["_meta"]["n_requests"]
            comp_tag_ = f"COMPACTED×{n_comp_}" if n_comp_ else "no-compact"

            out_ = {
                "session_id": bundle_["session_id"],
                "meta": bundle_["_meta"],
                "tools": bundle_["tools"],
                "messages": _strip_internal_fields(raw_msgs),
            }
            (out_dir / f"{safe_}.json").write_text(
                json.dumps(out_, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            print(
                f"{indent}{safe_:48s}  reqs={n_req_:3d}  segs={n_seg_}  "
                f"{comp_tag_:14s}  msgs={len(raw_msgs):4d}{extra_tail}"
            )
            return len(raw_msgs), n_comp_

        # ── main trajectory
        ops_tag = bundle["_meta"]["opsname"] or "(no opsname)"
        main_msgs, main_n_comp = _emit(
            bundle, main_safe_name, indent="  ",
            extra_tail=f"  sid={short_sid}  ops={ops_tag}",
        )
        summary["written"] += 1
        summary["total_msgs"] += main_msgs
        if main_n_comp:
            summary["sessions_with_compaction"] += 1
            summary["total_compactions"] += main_n_comp

        # ── subagent trajectories (opt-in)
        if args.export_subagents:
            sub_bundles = build_subagent_trajectories(conn, s)
            sub_count = len(sub_bundles)
            for sub_idx, (acid, sub_type, sub_bundle) in enumerate(sub_bundles, start=1):
                sub_type_safe = _FNAME_SAFE_RE.sub("_", sub_type or "subagent").strip("._-") or "subagent"
                suffix = f"sub{sub_idx}" if sub_count > 1 else "sub"
                sub_safe = f"{main_safe_name}__{suffix}_{sub_type_safe}"
                tail = f"  acid={acid[:8]}  type={sub_type}"
                s_msgs, s_comp = _emit(
                    sub_bundle, sub_safe, indent="    └─ ", extra_tail=tail
                )
                summary["subagents_written"] += 1
                summary["subagent_msgs"] += s_msgs
                if s_comp:
                    summary["total_compactions"] += s_comp

    print()
    print(f"sessions:  {summary['written']}/{summary['total_sessions']} written "
          f"(skipped {summary['skipped_no_main']} with no main thread)")
    if args.name_by == "opsname":
        print(f"naming:    {summary['named_by_opsname']} by opsname, "
              f"{summary['named_by_session_fallback']} fell back to session_id")
    if args.export_subagents:
        print(f"subagents: {summary['subagents_written']} written  "
              f"(msgs={summary['subagent_msgs']})")
    print(f"compact:   {summary['sessions_with_compaction']} session(s) with /compact "
          f"({summary['total_compactions']} compactions across main+sub)")
    print(f"main msgs: {summary['total_msgs']}")
    print(f"out:       {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
