"""Normalize 140 OpenAI-format JSON files into a single JSONL with a uniform
schema, so HuggingFace datasets doesn't have to merge schemas across files.

Each message ends up with exactly four fields:
  role          : str                    (required)
  content       : str                    (always present, "" if originally null/missing)
  tool_calls    : list[{id, type, function:{name, arguments}}]  (always present, [] if missing)
  tool_call_id  : str                    (always present, "" if missing)

`function.arguments` is always stringified (json.dumps if it was a dict).
This eliminates the cross-file struct/list/null type inference mismatch that
HuggingFace `cast_table_to_schema` was choking on.
"""
import argparse, glob, json, os, sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__)) + "/one_shot_success"
DST = os.path.dirname(os.path.abspath(__file__)) + "/one_shot_success.jsonl"

def _norm_args(a):
    if a is None:
        return ""
    if isinstance(a, str):
        return a
    # arguments came in as a real dict/list -> serialize back to string so all
    # rows share the same Arrow type.
    return json.dumps(a, ensure_ascii=False)


def _norm_tool_call(tc):
    fn = tc.get("function") or {}
    return {
        "id": str(tc.get("id") or ""),
        "type": str(tc.get("type") or "function"),
        "function": {
            "name": str(fn.get("name") or ""),
            "arguments": _norm_args(fn.get("arguments")),
        },
    }


def _norm_message(m):
    tcs = m.get("tool_calls")
    if not isinstance(tcs, list):
        tcs = []
    return {
        "role": str(m.get("role") or ""),
        "content": "" if m.get("content") is None else str(m.get("content")),
        "tool_calls": [_norm_tool_call(tc) for tc in tcs],
        "tool_call_id": "" if m.get("tool_call_id") is None else str(m.get("tool_call_id")),
    }


def main():
    ap = argparse.ArgumentParser(
        description="把目录下所有 OpenAI 格式 *.json 归一化合并成单个 JSONL。")
    ap.add_argument("src_dir", nargs="?", default=SRC_DIR,
                    help=f"输入目录（递归收集 *.json）。默认 {SRC_DIR}")
    ap.add_argument("--out", default=DST,
                    help=f"输出 JSONL 路径。默认 {DST}")
    args = ap.parse_args()

    src_dir = args.src_dir
    dst = args.out
    if not os.path.isdir(src_dir):
        sys.exit(f"输入目录不存在: {src_dir}")

    # 递归收集，覆盖增强产物的子目录结构
    files = sorted(glob.glob(os.path.join(src_dir, "**", "*.json"), recursive=True))
    print(f"normalizing {len(files)} files -> {dst}")

    out_dir = os.path.dirname(os.path.abspath(dst))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_rows = 0
    n_msgs_total = 0
    with open(dst, "w", encoding="utf-8") as out:
        for f in files:
            with open(f, encoding="utf-8") as fh:
                obj = json.load(fh)
            msgs = obj.get("messages") or []
            if not isinstance(msgs, list):
                print(f"  WARN: {f}: messages is {type(msgs).__name__}, skipping")
                continue
            row = {"messages": [_norm_message(m) for m in msgs]}
            n_msgs_total += len(row["messages"])
            n_rows += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"done. wrote {n_rows} rows, {n_msgs_total} messages total.")
    print(f"size: {os.path.getsize(dst) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
