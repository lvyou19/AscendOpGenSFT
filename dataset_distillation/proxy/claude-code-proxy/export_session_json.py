#!/usr/bin/env python3
"""导出 trace.db 中的 session 列表到文本文件."""

import sqlite3
import os
import argparse
import json

DEFAULT_DB_PATH = '/home/l00868164/agent/claude-code-proxy/cc_traces/trace.db'
DEFAULT_OUTPUT = '/home/l00868164/agent/db_log/session_list.txt'


def get_first_request_user_prompt(cursor, session_id):
    """获取 session 第一个请求中 block[2] 的用户指令文本."""
    cursor.execute('''
        SELECT request_body_json
        FROM requests
        WHERE session_id = ?
        ORDER BY started_at
        LIMIT 1
    ''', (session_id,))
    row = cursor.fetchone()
    if not row or not row[0]:
        return ''

    try:
        body = json.loads(row[0])
    except json.JSONDecodeError:
        return ''

    messages = body.get('messages', [])
    if not messages:
        return ''

    first_msg = messages[0]
    content = first_msg.get('content', '')
    if isinstance(content, list) and len(content) > 2:
        block = content[2]
        if block.get('type') == 'text':
            return block.get('text', '')
    elif isinstance(content, str):
        return content

    return ''


def _dump_one_session_full(cursor, session_id):
    """Dump one session's complete content into the test-file shape:
       {session, requests, agent_calls, messages, tool_events}.

    JSON-text columns in `requests` / `agent_calls` are re-parsed back into
    dict/list so the resulting file is human-readable and round-trips through
    scripts/export_training.py --db <file>.json without re-serialization."""
    sess_row = cursor.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not sess_row:
        return None
    reqs = [dict(r) for r in cursor.execute(
        "SELECT * FROM requests WHERE session_id = ? ORDER BY started_ms ASC",
        (session_id,),
    ).fetchall()]
    acs = [dict(r) for r in cursor.execute(
        "SELECT * FROM agent_calls WHERE session_id = ? ORDER BY started_ms ASC",
        (session_id,),
    ).fetchall()]
    # messages and tool_events may or may not exist depending on schema version
    msgs = []
    tes = []
    try:
        msgs = [dict(r) for r in cursor.execute(
            "SELECT * FROM messages WHERE trace_id IN "
            "(SELECT trace_id FROM requests WHERE session_id = ?) "
            "ORDER BY trace_id, msg_index",
            (session_id,),
        ).fetchall()]
    except sqlite3.OperationalError:
        pass
    try:
        tes = [dict(r) for r in cursor.execute(
            "SELECT * FROM tool_events WHERE session_id = ? ORDER BY event_time_ms",
            (session_id,),
        ).fetchall()]
    except sqlite3.OperationalError:
        pass

    json_columns = (
        "request_body_json", "converted_request_json", "response_body_json",
        "error_json", "headers_json", "prefix_hashes_json", "tool_names_json",
        "response_tool_uses_json", "response_usage_json", "extra_json",
        "child_request_ids_json", "history_request_ids_json",
        "full_content_json", "block_types_json", "raw_block_json",
    )
    for collection in (reqs, acs, msgs, tes):
        for r in collection:
            for k in json_columns:
                v = r.get(k)
                if isinstance(v, str):
                    try:
                        r[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass

    return {
        "session": dict(sess_row),
        "requests": reqs,
        "agent_calls": acs,
        "messages": msgs,
        "tool_events": tes,
    }


def _sanitize_for_filename(s):
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]+", "_", s or "").strip("._-") or "session"


def main():
    parser = argparse.ArgumentParser(description='导出 trace.db session 列表 / 单 session 完整内容')
    parser.add_argument('-i', '--input', default=DEFAULT_DB_PATH, help=f'输入数据库路径 (默认: {DEFAULT_DB_PATH})')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT, help='输出文件 (--full 时为输出目录)')
    parser.add_argument('--json', action='store_true', help='以 JSON 格式输出 session 列表')
    parser.add_argument('--full', action='store_true',
                        help='把每个 session 的完整内容(session/requests/agent_calls/messages/'
                             'tool_events)各导出为 <output_dir>/<session_id>.json,可被 '
                             'scripts/export_training.py 直接消费')
    parser.add_argument('--session', action='append',
                        help='--full 模式下只导指定 session_id(可重复传入)')
    args = parser.parse_args()

    conn = sqlite3.connect(args.input)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if args.full:
        os.makedirs(args.output, exist_ok=True)
        if args.session:
            placeholders = ",".join("?" * len(args.session))
            rows = cursor.execute(
                f"SELECT session_id FROM sessions WHERE session_id IN ({placeholders}) "
                f"ORDER BY first_seen DESC",
                tuple(args.session),
            ).fetchall()
        else:
            rows = cursor.execute(
                "SELECT session_id FROM sessions ORDER BY first_seen DESC"
            ).fetchall()
        n = 0
        for r in rows:
            sid = r["session_id"]
            payload = _dump_one_session_full(cursor, sid)
            if not payload:
                continue
            fname = f"{_sanitize_for_filename(sid)}.json"
            path = os.path.join(args.output, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            n += 1
            print(f"  wrote {path}  (requests={len(payload['requests'])}, "
                  f"agent_calls={len(payload['agent_calls'])})")
        conn.close()
        print(f"已导出 {n} 个 session 到 {args.output}")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    cursor.execute('''
        SELECT
            s.session_id,
            s.origin,
            s.first_seen,
            s.last_seen,
            s.request_count,
            s.cc_version,
            COUNT(DISTINCT r.trace_id) as request_count_actual
        FROM sessions s
        LEFT JOIN requests r ON s.session_id = r.session_id
        GROUP BY s.session_id
        ORDER BY s.first_seen DESC
    ''')

    sessions = [dict(row) for row in cursor.fetchall()]

    if args.json:
        output_data = []
        for s in sessions:
            prompt = get_first_request_user_prompt(cursor, s['session_id'])
            output_data.append({
                'session_id': s['session_id'],
                'origin': s['origin'],
                'first_seen': s['first_seen'],
                'last_seen': s['last_seen'],
                'request_count': s['request_count_actual'],
                'cc_version': s['cc_version'],
                'user_prompt': prompt,
            })
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    else:
        with open(args.output, 'w') as f:
            f.write(f"共有 {len(sessions)} 个 session\n\n")
            for s in sessions:
                sid = (s['session_id'] or '')[:38]
                origin = (s['origin'] or '')[:18]
                first = (s['first_seen'] or '')[:20]
                reqs = str(s['request_count_actual'])
                ver = s['cc_version'] or ''

                f.write(f"{'=' * 80}\n")
                f.write(f"session_id: {sid}\n")
                f.write(f"origin:     {origin}\n")
                f.write(f"first_seen: {first}\n")
                f.write(f"requests:   {reqs}\n")
                f.write(f"cc_version: {ver}\n")

                prompt = get_first_request_user_prompt(cursor, s['session_id'])
                if prompt:
                    f.write(f"\n用户指令:\n{prompt}\n")
                else:
                    f.write("\n用户指令: (无)\n")
                f.write("\n")

    conn.close()
    print(f"已写入 {args.output}")


if __name__ == '__main__':
    main()
