# `tools/` · 工具清单（2026-10-20 补入的三件）

> 来源：`E:\infer-build\kv-stream\` 开发树同字节拷贝（未改内容；逐件 SHA256 相同）。
> 用途：补上 `docs/修复清单-20261019.md` §1.4 提到的 `agg_gateway.py` 与 8 GB 档相关脚本，使修复项可从仓内复核/复现。
> 每行只写脚本自带 usage/argparse 里真实存在的参数（不编 flag）。

| 文件 | 干什么 | 谁用 | 需引擎在跑？ | 入口命令（取自脚本 argparse） |
|---|---|---|---|---|
| `agg_gateway.py` | 超池题面的分片/聚合网关：按整结构单元（空行分段）打包、按池容量定片数、`--max-shard-tokens` 封顶；单个结构单元 > 池时走 `unit_broken` 拒答（`sum_method="unit_broken"`、`low_confidence=true`），不回落模型 reduce。附 `--selftest`（`partition_exact`、`unit_broken` 等离线自检） | 修复 1.4 的复核者；超池题面验收 | **是**：`--selftest` 除外（其余要 POST 到 `--base-url` 的 `/v1/chat/completions`） | `python tools/agg_gateway.py --prompt-file <材料文件> --question <问题> --pool-tokens <池 token 数> [--max-shard-tokens 32768] [--base-url http://127.0.0.1:8080] [--model default]`；离线自检：`python tools/agg_gateway.py --selftest` |
| `kv_plan.py` | 27B ternary 的 KV 池规划（纯算术）：按 `--vram-gib` / `--quant` / `--context` 算池页数与 engine MiB；带 `PAGE_BYTES` 表（`k8v4` 1,646,592 B / `nvfp4` 1,179,648 B / `rk2v4-e8` 884,736 B）与 `--selftest` | 8 GB/小显存档配参者；`--host-kv-mib` 下界核算 | **否**：纯算术（stdlib only） | `python tools/kv_plan.py --vram-gib 16 --quant ptq1 --context 262144 [--host-kv-mib 2816] [--desktop-mib 300] [--reserve-mib 500]`；自检：`python tools/kv_plan.py --selftest` |
| `accept_8g.py` | 8 GiB 档端到端验收：6 个 streaming case 走聊天网关，用引擎日志、网关日志、GPU 采样与主机空闲内存判 GREEN/RED；退出码 0=GREEN、1=RED，不重试、不改参数 | 8 GB 档回归/交付验收 | **是**：需要网关与引擎在跑（默认 `--gateway http://127.0.0.1:8081`） | `python tools/accept_8g.py --gateway http://127.0.0.1:8081 --engine-log <引擎日志> --gateway-log <网关日志> --filler <filler.txt> [--pool 15616] [--engine-max-mib 7442] [--host-min-gib 20]` |

> 说明：`accept_8g.py` 的默认日志/filler 路径是开发机路径（`E:\infer-build\kv-stream\…`），换机时必须用上面的参数覆盖。
> 三件都是 stdin/stdout CLI，没有新增依赖（Python 标准库）。
