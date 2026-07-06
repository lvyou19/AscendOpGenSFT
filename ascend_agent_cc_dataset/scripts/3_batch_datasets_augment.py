"""
批量「自动识别 + 环境信息泛化增强」脚本。

针对 /home/l00899543/SFT_DATASETS/第二批/ 下每条 json（一个文件一条数据，含 __sub_ 子 agent 轨迹）。
不硬编码任何环境值：对每个文件用稳定锚点自动抽取它自己的环境身份，再整体替换为一套
随机但内部自洽的新身份。

解决两类问题（见 9_topk_diff_analysis.md）：
  1. 路径漂移：uid / 中间目录 / 项目根 / benchmark 属主在推理环境与训练环境不一致。
  2. 日期格式差异：currentDate 的值与分隔符（- 或 /）变化。

关键设计：
  - 双身份独立：技能仓库属主（cwd/memory）与 benchmark 数据属主常是不同的人，各自独立随机。
  - memory 路径正向派生：/home/{user}/{middle}cannbot-skills 把 '/' 和 '_' 都换成 '-' 加前导 '-'
    （有损编码，只能正向派生，不能反解析）。
  - 日期只按 '# currentDate' 锚点替换，绝不全局 replace（保护 auto-memory 指令里的示例日期）。
  - uid 不假设格式（lvyou 无数字、t00893162 以 t 开头），全靠路径锚点抽取。
  - 替换按 old 串长度降序，长串先替换避免短串破坏长串。

用法：
  python3 batch_augment_env.py SRC_DIR --out-dir OUT_DIR --copies 10 [--verify] [--limit-subdir lvyou]
"""

import argparse
import glob
import json
import os
import random
import re
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# 文本提取
# ---------------------------------------------------------------------------
def message_strings(msg):
    """返回 (kind, value) 列表：kind in {'content','tool_calls'}，便于回写。"""
    parts = []
    c = msg.get('content')
    if isinstance(c, str):
        parts.append(('content', c))
    if msg.get('tool_calls'):
        parts.append(('tool_calls', json.dumps(msg['tool_calls'], ensure_ascii=False)))
    return parts


def full_text(data):
    out = []
    for m in data['messages']:
        for _, s in message_strings(m):
            out.append(s)
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# 自动识别：抽取该文件的环境身份
# ---------------------------------------------------------------------------
CWD_RE = re.compile(r'Primary working directory: /home/([^/\n]+)/((?:[^/\n]+/)*?)cannbot-skills')
HOME_USER_RE = re.compile(r'/home/([^/\s"\\,)，`]+)/')
# 任意位置出现的 cannbot-skills 仓库根（含可空中间目录），用于在没有 cwd 锚点的子 agent 文件里也能替换
REPO_RE = re.compile(r'/home/([^/\s"\\,)，`]+)/((?:[A-Za-z0-9_]+/)*?)cannbot-skills')
# 任意位置出现的 memory 派生串
MEM_RE = re.compile(r'(-home-[A-Za-z0-9-]+?-cannbot-skills)')
DATE_ANCHOR_RE = re.compile(r"(# currentDate\nToday's date is )(\d{4})([-/])(\d{2})[-/](\d{2})(\.)")
NPU_RE = re.compile(r'npu=(\d+)')

# /home/{user}/{seg1}/ 的一级子目录名。除保留约定外，都是输出/工作目录命名（位置信息），需随机化。
HOME_SEG1_RE = re.compile(r'/home/[^/\s"\\,)，`]+/([A-Za-z0-9_]+)/')
# 一级子目录里的固定约定（不随机化）：基准库目录、技能仓库
KEEP_SEG1 = {'benchmarks', 'cannbot-skills'}
# 输出根下、算子目录(\d+_Op)之前的「run 命名段」。如 ops_benchmark_result/35b_..._pass2/18_Index/
# 仅匹配紧邻算子目录前的那一段，避免误伤算子目录本身。
RUN_NAME_RE = re.compile(r'/([A-Za-z0-9_.]+)/\d+_[A-Za-z0-9]+/')


def extract_identity(data):
    text = full_text(data)

    # 技能仓库属主 + 中间目录（可能没有，子 agent 文件就没有）
    skills_owner = None
    skills_middle = ''
    m = CWD_RE.search(text)
    if m:
        skills_owner = m.group(1)
        skills_middle = m.group(2)  # 形如 'Collect_trace/' 或 ''

    # 所有出现在 /home/X/ 里的 user 集合（含 benchmark 属主、输出目录属主等）
    home_users = set(HOME_USER_RE.findall(text))

    # currentDate
    dm = DATE_ANCHOR_RE.search(text)
    cur_date = (dm.group(2), dm.group(3), dm.group(4), dm.group(5)) if dm else None

    # npu
    npus = set(NPU_RE.findall(text))

    # 所有 cannbot-skills 仓库根（owner, middle），覆盖无 cwd 锚点的子 agent 文件
    repos = set(REPO_RE.findall(text))  # {(owner, middle)}

    # 技能仓库中间目录的各段（如 'Collect_trace/' -> {'Collect_trace'}）。
    # 这些段由 repo 规则负责替换、且必须与 memory 串同步派生，
    # 不能再被当成独立命名段二次替换（否则 cwd 与 memory 脱节，verify 报 memory mismatch）。
    repo_middle_segs = set()
    for _owner, mid in repos:
        for seg in mid.split('/'):
            if seg:
                repo_middle_segs.add(seg)
    if skills_middle:
        for seg in skills_middle.split('/'):
            if seg:
                repo_middle_segs.add(seg)

    # 输出/工作目录命名段（位置信息，需随机化）：
    #   一级子目录名（排除 benchmarks/cannbot-skills 固定约定，以及技能仓库中间目录段）
    seg1 = {s for s in HOME_SEG1_RE.findall(text)
            if s not in KEEP_SEG1 and s not in repo_middle_segs}
    #   输出根下、算子目录之前的 run 命名段（同样排除技能仓库中间目录段）
    run_names = {s for s in RUN_NAME_RE.findall(text) if s not in repo_middle_segs}
    naming_segs = seg1 | run_names

    return {
        'skills_owner': skills_owner,
        'skills_middle': skills_middle,
        'repos': repos,
        'home_users': home_users,
        'cur_date': cur_date,
        'npus': npus,
        'naming_segs': naming_segs,
    }


# ---------------------------------------------------------------------------
# 采样新身份
# ---------------------------------------------------------------------------
UID_POOLS = [
    lambda: f"l{random.randint(10000000, 99999999)}",
    lambda: f"t{random.randint(10000000, 99999999)}",
    lambda: f"z{random.randint(10000000, 99999999)}",
    lambda: random.choice(['devuser', 'opsgen', 'kdev', 'aiteam']),
]
MIDDLE_POOL = ['', 'Collect_trace/', 'workspace/projects/', 'Ascend_evaluation/', 'eval_runs/']
# 输出/工作目录命名段的随机词池（位置命名，纯换名）
NAMING_POOL = [
    'ops_result', 'eval_output', 'run_results', 'bench_out', 'result_bak',
    'sft_eval', 'kernel_runs', 'gen_output', 'baseline_runs', 'op_results',
]


def new_naming_seg():
    # 偶尔拼一个带随机后缀的 run 名，模拟 35b_..._passN 这类命名
    if random.random() < 0.4:
        tag = random.choice(['7b', '35b', '70b', 'v1', 'v2', 'kimi', 'local'])
        suffix = random.choice(['pass1', 'pass2', 'compact', 'baseline', 'sft'])
        return f"{tag}_{random.choice(NAMING_POOL)}_{suffix}"
    return random.choice(NAMING_POOL)


def new_uid():
    return random.choice(UID_POOLS)()


def derive_memory_dir(user, middle):
    """/home/{user}/{middle}cannbot-skills -> -home-{user}-...-cannbot-skills (/ 和 _ 都转 -)"""
    repo = f"home/{user}/{middle}cannbot-skills"
    return '-' + repo.replace('/', '-').replace('_', '-')


def sample_mapping(identity):
    # 收集所有需要映射的旧 user：home 路径里出现的 + 各仓库属主
    old_users = set(identity['home_users'])
    old_users |= {owner for owner, _ in identity['repos']}
    if identity['skills_owner']:
        old_users.add(identity['skills_owner'])

    new_for = {}
    used = set()
    for u in sorted(old_users):
        while True:
            nu = new_uid()
            if nu not in used and nu != u:
                break
        used.add(nu)
        new_for[u] = nu

    # 中间目录：对每个 (old_user, old_middle) 确定性映射到一个新中间目录，
    # 保留「有/无」形态特征，保证 cwd 与 memory 串派生一致。
    # 关键：新中间目录的任一段都不能与本文件的「命名段」旧值相同——否则命名段替换
    # 规则（/seg/ 边界）会在 repo 替换之后二次改写 cwd 里的中间目录，而 memory 串
    # （横线形式）不被同步改写，导致 cwd 与 memory 脱节（verify memory mismatch）。
    naming_olds = set(identity.get('naming_segs') or set())

    def _no_collision(mid):
        return all(seg not in naming_olds for seg in mid.split('/') if seg)

    middle_cache = {}

    def middle_for(old_user, old_mid):
        key = (old_user, old_mid)
        if key not in middle_cache:
            if old_mid:
                cands = [x for x in MIDDLE_POOL if x and _no_collision(x)]
            else:
                cands = [x for x in ['', '', 'Collect_trace/'] if _no_collision(x)]
            # 全被过滤掉时退回空中间目录（绝不与命名段冲突）
            middle_cache[key] = random.choice(cands) if cands else ''
        return middle_cache[key]

    # 日期：值随机；~20% 切换分隔符
    new_date = None
    if identity['cur_date']:
        base = date(2025, 1, 1) + timedelta(days=random.randint(0, 900))
        sep = random.choice(['-', '/']) if random.random() < 0.2 else identity['cur_date'][1]
        new_date = (str(base.year), f"{base.month:02d}", sep, f"{base.day:02d}")

    # npu：每个旧 npu 映射到一个新 npu
    npu_map = {n: str(random.randint(0, 15)) for n in identity['npus']}

    # 输出/工作目录命名段：每个旧命名段映射到一个新随机命名（去重，避免撞名）
    naming_map = {}
    used_names = set()
    for seg in sorted(identity['naming_segs'], key=lambda s: -len(s)):
        while True:
            ns = new_naming_seg()
            if ns not in used_names and ns != seg:
                break
        used_names.add(ns)
        naming_map[seg] = ns

    dirty = random.random() < 0.2

    return {
        'user_map': new_for,
        'middle_for': middle_for,
        'new_date': new_date,
        'npu_map': npu_map,
        'naming_map': naming_map,
        'dirty': dirty,
    }


# ---------------------------------------------------------------------------
# 构造替换对（长度降序）
# ---------------------------------------------------------------------------
def build_replacements(identity, mapping):
    pairs = []

    # 所有 cannbot-skills 仓库根 + 派生 memory 串（覆盖无 cwd 锚点的子 agent 文件）。
    # 同一 (owner, middle) 的中间目录用确定性新值，保证 cwd 与 memory 串派生一致。
    for old_user, old_mid in sorted(identity['repos']):
        new_user = mapping['user_map'][old_user]
        new_mid = mapping['middle_for'](old_user, old_mid)
        old_repo = f"/home/{old_user}/{old_mid}cannbot-skills"
        new_repo = f"/home/{new_user}/{new_mid}cannbot-skills"
        pairs.append((old_repo, new_repo))
        pairs.append((derive_memory_dir(old_user, old_mid),
                      derive_memory_dir(new_user, new_mid)))

    # 每个 home user：裸 /home/{user}/ 兜底替换（短串，最后）
    user_pairs = []
    for old_user, new_user in mapping['user_map'].items():
        user_pairs.append((f"/home/{old_user}/", f"/home/{new_user}/"))
    pairs.extend(user_pairs)

    # npu
    for old_n, new_n in mapping['npu_map'].items():
        pairs.append((f"npu={old_n}", f"npu={new_n}"))

    # 输出/工作目录命名段（位置信息）：按 /seg/ 边界替换，避免子串误伤。
    # benchmarks/NPUKernelBench/level1 等固定约定已在抽取阶段排除，不在此处。
    for old_seg, new_seg in mapping['naming_map'].items():
        pairs.append((f"/{old_seg}/", f"/{new_seg}/"))

    # 算子编号前缀（如 9_TopK 的 "9_"）属于算子语义标识，必须保持不变，不做替换。

    # 长串优先
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


def apply_text(text, pairs, mapping):
    for old, new in pairs:
        if old and old != new:
            text = text.replace(old, new)

    # 日期：仅按锚点替换（保护指令示例日期）
    if mapping['new_date']:
        y, mo, sep, dd = mapping['new_date']
        text = DATE_ANCHOR_RE.sub(
            lambda m: f"{m.group(1)}{y}{sep}{mo}{sep}{dd}{m.group(6)}", text
        )

    # 脏数据注入：把 benchmark 路径的 /level1 段改成 //level1（若尚未是 //）
    if mapping['dirty']:
        text = re.sub(r'(?<!/)/level1', '//level1', text)
    return text


def apply_message(msg, pairs, mapping):
    out = dict(msg)
    if isinstance(out.get('content'), str):
        out['content'] = apply_text(out['content'], pairs, mapping)
    if out.get('tool_calls'):
        raw = json.dumps(out['tool_calls'], ensure_ascii=False)
        out['tool_calls'] = json.loads(apply_text(raw, pairs, mapping))
    return out


def augment(data, identity, mapping):
    pairs = build_replacements(identity, mapping)
    return {**{k: v for k, v in data.items() if k != 'messages'},
            'messages': [apply_message(m, pairs, mapping) for m in data['messages']]}


# ---------------------------------------------------------------------------
# 自洽性验证
# ---------------------------------------------------------------------------
def verify(aug_data, identity, mapping, src):
    text = full_text(aug_data)
    errs = []
    # 1) 旧 user 零泄漏
    for old_user in mapping['user_map']:
        if f"/home/{old_user}/" in text:
            errs.append(f"old user leaked: {old_user}")
    # 2) memory 与 cwd 派生一致
    if identity['skills_owner']:
        m = CWD_RE.search(text)
        mem = re.search(r'/root/\.claude/projects/(-home-[^/\s"\\]+?-cannbot-skills)/memory', text)
        if m and mem:
            expect = derive_memory_dir(m.group(1), m.group(2))
            if expect != mem.group(1):
                errs.append(f"memory mismatch: cwd-> {expect} vs mem {mem.group(1)}")
    # 3) 日期锚点已替换
    if mapping['new_date']:
        dm = DATE_ANCHOR_RE.search(text)
        if dm:
            y, mo, sep, dd = mapping['new_date']
            if (dm.group(2), dm.group(4), dm.group(5)) != (y, mo, dd):
                errs.append("date anchor not applied")
    # 4) JSON 可序列化 & messages 数一致由调用方保证
    return errs


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src_dir')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--copies', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--limit-subdir', default=None, help='只处理某个子目录（干跑用）')
    ap.add_argument('--exclude-suffix', action='append', default=[],
                    metavar='SUFFIX',
                    help='排除名字（小写后）以该后缀结尾的目录，整树跳过。可重复。'
                         '如 --exclude-suffix failed 排除 level_1_failed/。')
    args = ap.parse_args()

    random.seed(args.seed)
    pattern = os.path.join(args.src_dir, '**', '*.json')
    files = sorted(glob.glob(pattern, recursive=True))
    if args.limit_subdir:
        files = [f for f in files if f'/{args.limit_subdir}/' in f or
                 os.path.relpath(f, args.src_dir).startswith(args.limit_subdir + os.sep)]
    # 排除以指定后缀结尾的目录（路径任一层匹配即整条排除）
    suffixes = [s.lower() for s in args.exclude_suffix]
    if suffixes:
        def _excluded(path):
            rel = os.path.relpath(path, args.src_dir)
            for part in rel.split(os.sep)[:-1]:  # 只看目录层，不含文件名
                pl = part.lower()
                if any(pl.endswith(s) for s in suffixes):
                    return True
            return False
        files = [f for f in files if not _excluded(f)]

    total_out = 0
    fail = 0
    verify_errs = 0
    for f in files:
        try:
            data = json.load(open(f, encoding='utf-8'))
        except Exception as e:
            print(f"[SKIP] {f}: {e}")
            fail += 1
            continue
        if 'messages' not in data:
            print(f"[SKIP] {f}: no messages")
            fail += 1
            continue

        # 删除不需要的顶层字段（augment 会原样透传除 messages 外的顶层键，
        # 在此处剥离即可让增强产物也不含这些字段）
        for k in ('session_id', 'meta', 'tools'):
            data.pop(k, None)

        rel = os.path.relpath(f, args.src_dir)
        stem = os.path.basename(f)
        for suf in ('.raw.json', '.json'):
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
                break
        out_subdir = os.path.join(args.out_dir, os.path.dirname(rel))
        os.makedirs(out_subdir, exist_ok=True)

        identity = extract_identity(data)
        for i in range(args.copies):
            mapping = sample_mapping(identity)
            aug = augment(data, identity, mapping)
            if args.verify:
                errs = verify(aug, identity, mapping, f)
                if errs:
                    verify_errs += 1
                    print(f"[VERIFY-FAIL] {rel} aug{i:02d}: {errs}")
            assert len(aug['messages']) == len(data['messages'])
            json.dump(aug, open(os.path.join(out_subdir, f"{stem}_aug{i:02d}.json"),
                      'w', encoding='utf-8'), ensure_ascii=False, indent=2)
            total_out += 1

    print(f"\nfiles processed: {len(files)}  failed: {fail}  outputs: {total_out}  "
          f"verify_errs: {verify_errs}")


if __name__ == '__main__':
    main()
