"""
隔离 A/B（臂 B）：同 engine、同权重、同 argv，**只改 --kv-capacity**，
在**独立端口 8090** 起一份 ninfer-ptq1，用来验证"把 KV 池放大"能不能让 20.7K 提示词不再撞 host-restore。

- 参数来源 = `tools_dashboard\ninfer_ctl.py` 的 TIERS/ENV（**逐字复用，不手抄**，避免转录错误）。
- 只覆盖三处：`--port`、`--kv-capacity`、日志/进程标识。
- 不碰生产档位（1249/8087）、不写任何生产文件。
用法：
  python _kvpool_ab_start.py --pages 420 [--port 8090]
打印（供落盘）：engine exe、model 路径与大小、逐字 argv、env、pid。
"""
import argparse
import importlib.util
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOME = r"C:\Users\wo739\Documents\ai"
TD = os.path.join(HOME, "tools_dashboard")
CTL = os.path.join(TD, "ninfer_ctl.py")

spec = importlib.util.spec_from_file_location("ninfer_ctl", CTL)
ctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctl)          # 该模块只在 main() 里动手，import 无副作用

ap = argparse.ArgumentParser()
ap.add_argument("--tier", default="ptq1")
ap.add_argument("--pages", type=int, help="KV 池页数（64 token/页）")
ap.add_argument("--capacity", type=int, help="直接给 KV 池 token 数（与 --pages 二选一）")
ap.add_argument("--window", type=int, help="NINFER_KV_WINDOW（覆盖默认 16384）")
ap.add_argument("--no-spec", action="store_true", help="去掉 --spec/--draft-tokens（省 0.42 GiB 给池子）")
ap.add_argument("--port", type=int, default=8090)
ap.add_argument("--logtag", default="kvab")
args = ap.parse_args()
if args.capacity is None and args.pages is None:
    raise SystemExit("需要 --pages 或 --capacity")
cap_tokens = args.capacity if args.capacity is not None else args.pages * 64

tier = args.tier
info = ctl.TIERS[tier]
models_dir = ctl.MODELS
model_path = os.path.join(models_dir, info["model"])

# CLI 契约实测（本引擎）：**模型必须紧跟 exe、在所有开关之前**
#   ninfer-serve.exe <model.ninfer> [options]
#   反例：把 --host/--port 放在模型前面 ⇒ `unknown argument: <模型路径>`（本轮踩过两次）
vals = [ctl.ENGINE, model_path] + list(ctl.COMMON) + ["--model-id", info["model_id"]] + list(info["argv"])


def override(flag, value):
    """同一开关可能在本档 argv 里出现多次（COMMON 与档位 argv 各一份），全部改掉。"""
    n = vals.count(flag)
    if n == 0:
        vals.extend([flag, str(value)])
        return
    out, i = [], 0
    while i < len(vals):
        if vals[i] == flag:
            out.append(flag)
            out.append(str(value))
            i += 2
        else:
            out.append(vals[i])
            i += 1
    vals[:] = out
    print("NOTE: 覆盖 %s 共 %d 处 -> %s" % (flag, n, value))


override("--port", args.port)                 # 换成隔离端口
override("--kv-capacity", cap_tokens)         # ← 本实验唯一变量（或变量之一）

# 可选：关投机（官方：「要池就不要开投机」——省 0.42 GiB 给池子）
if args.no_spec:
    for flag in ("--spec", "--draft-tokens", "--lm-head-draft"):
        while flag in vals:
            i = vals.index(flag)
            del vals[i:i + 2]
    print("NOTE: 已移除 --spec / --draft-tokens / --lm-head-draft（关投机）")

# 可选：改常驻窗口（决定池页预算 = window/64 + prefill/64 + 8）
extra_env = {}
if args.window is not None:
    extra_env["NINFER_KV_WINDOW"] = str(args.window)
    print("NOTE: NINFER_KV_WINDOW -> %d" % args.window)

env = dict(os.environ)
env.update(ctl.ENV)
env.update(info.get("env", {}))

engine = ctl.ENGINE
out_log = os.path.join(TD, "infer-%s.out.log" % args.logtag)
err_log = os.path.join(TD, "infer-%s.err.log" % args.logtag)

print("ENGINE   =", engine, os.path.getsize(engine) if os.path.exists(engine) else "MISSING")
print("MODEL    =", model_path, os.path.getsize(model_path) if os.path.exists(model_path) else "MISSING")
print("CAPACITY =", cap_tokens, "tokens（= %d 页）" % (cap_tokens // 64))
print("PORT     =", args.port)
print("ARGV     =", " ".join(vals))
print("ENV      =", {k: env[k] for k in sorted(ctl.ENV) })
print("OUT/ERR  =", out_log, err_log)

fo = open(out_log, "ab")
fe = open(err_log, "ab")

# 只用 cmd 的重定向（`>log 2>&1`）—— 本机踩过：把 OS 文件句柄交给 Popen 会让引擎拿不到 stderr。
# env 用 `set K=V && ...` 前缀（cmd 内联设置）；命令行里不能有引号（会与 WMI 的引号打架），我们的路径都不含空格 ✅
env_prefix = " && ".join(["set %s=%s" % (k, env[k]) for k in sorted(ctl.ENV)] + ["set NINFER_TERNARY_PTQ1_FAST=1"])
for k, v in extra_env.items():
    env_prefix += " && set %s=%s" % (k, v)
print("WINDOW   =", extra_env.get("NINFER_KV_WINDOW", ctl.ENV.get("NINFER_KV_WINDOW")), "（未覆盖时 = 启动器默认）")
cmdline = "cmd.exe /c " + env_prefix + " && " + " ".join(vals) + " >" + out_log + " 2>&1"

# 发射交给 pwsh 走 WMI（本机铁律：常驻服务只能 WMI / wscript 起 —— 从 agent 的 shell 起会被连带杀掉；
# 另外本机 python 子进程调 WMI 会被拒（Invoke-CimMethod: 拒绝访问)）
cmd_file = os.path.join(TD, "._kvab_cmdline.txt")
with open(cmd_file, "w", encoding="utf-8") as f:
    f.write(cmdline)
print("CMDLINE  =", cmdline)
print("CMDFILE  =", cmd_file)
print("NEXT     = pwsh -File tmpwork\\_wmi_fire.ps1 -CmdlineFile <%s> -LogFile <%s>" % (cmd_file, err_log))

