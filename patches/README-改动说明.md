# patches · 引擎侧改动的逐文件清单与说明

本目录是**我们对上游引擎的改动**。全部改动**只碰引擎源码**，不含模型权重、不含二进制。

---

## 1. 基线（Baseline）——**先确认这一条，否则补丁打不上**

| 项 | 值 |
|---|---|
| 上游 | **NInfer** —— 原始仓 [`Neroued/ninfer`](https://github.com/Neroued/ninfer)（`master`，Apache-2.0）；基线分支同 [`ashalliants/ninfer-3090`](https://github.com/ashalliants/ninfer-3090) 的 `master`（本机存档目录名 `upstream-iamwavecut-ninfer-3090-0.11.0-rtx3090`）|
| 版本 | **`VERSION = 0.11.0-rtx3090`** |
| 本机完整快照 | `refs\infer-all-full\ninfer-all-master\`（2,455 件 / 71.9 MB；排除 `__pycache__` 后 2,419 件）|
| 上游 URL | ✅ **已核实（2026-10-04）**：<https://github.com/Neroued/ninfer>（原始）· <https://github.com/ashalliants/ninfer-3090>（基线，`master`）· <https://github.com/iamwavecut/ninfer-3090>（存档对应，`master`）|

**我们的工作树**：本仓 `src-tree/fusion-engine-src/`（本机同源工作树 `fusion-master\src\`，2,463 件，排除 `__pycache__`）。本目录下的 `changed-files\` 就是工作树里那批改动文件的**完整副本**，
路径与工作树一一对应 —— 直接把文件覆盖到你的上游树同名路径即可。

---

## 2. 这份清单是怎么来的（判据，可复现）

**用内容哈希逐文件比，不用时间戳**（时间戳一拷贝就全变，会把 2,400 个文件都报成"改过"）。

```
A（我们的树） = src-tree/fusion-engine-src\               2,463 件
B（上游基线） = refs\infer-all-full\ninfer-all-master\     2,455 件（排除 __pycache__ 后 2,419 件）

SAME    2,327      ← 逐字节相同。这一项同时证明"基线选对了"
CHANGED    92      ← 我方修改（逐文件 SHA256 比对）
ADDED      44      ← 我方新增（`src/ops/kvmem/` 主体、`src/ops/linear/t2/` PTQ1 件、`qw3` 策略层等；含 1 份误留的 .bak，见 §4）
REMOVED     0      ← 无：本树不缺上游任何文件
```

自己复核（对同一个基线跑，应当得到同样的 92 件修改 + 44 件新增）：

```bat
git diff --no-index --stat "<上游树>" "<你的树>"
```

机器可读清单：`changed-source.txt`（本目录收录的 38 件改动文件）｜完整对账：`changed-files.txt`（92 件修改 + 44 件新增）。

---

## 3. `changed-files/` 收录的 38 件，按主题分（与 `docs/方案/` 里的文档一一对应）

| # | 主题 | 文件 |
|---|---|---|
| 1 | **KVMem 环 / 存储解耦**（让"设备池 < 逻辑上下文"成为合法配置） | `src\core\paged_kv_cache.{cpp,h}` · `src\ops\kernel\paged_kv_address.cuh` · `src\models\qwen3_5\program\storage\context.cpp` · `…\program\storage\kv_store.h` · `…\state\decoder_state.{cpp,h}` |
| 2 | **host-backed 复用 + 惰性领用**（池装不下的检查点整段登记、按需物化） | `…\program\transactions\capture.cpp`（发布）· `…\program\planning\pressure.cpp`（采纳门）· `…\program\transactions\materialization.cpp`（有界恢复）|
| 3 | **按工作集定池 + 启动守卫** | `…\program\planning\startup.{cpp,h}` · `…\program\planning\request_plan.cpp` |
| 4 | **前缀身份 / 事务提交** | `…\program\prefix_identity.cpp` · `…\program\transactions\commit.cpp` · `…\program\prefill.cpp` |
| 5 | **程序主体接线** | `…\program\program.h` · `…\program\program_impl.{cpp,h}` |
| 6 | **显存 arena** | `src\core\arena.{cu,h}` |
| 7 | **按卡自适应档位**（本机 4080 SUPER 标定档并进内置表） | `src\runtime\engine\device_profiles.json` |
| 8 | **内核族** | `src\ops\softmax_attention\dense\causal_cache\`（9 件：`prompt_fp8`、`prompt_nvfp4`、`small_t_fp8`、`small_t_nvfp4`、`launch.h`）· `src\ops\linear\bf16\bf16_a16_tma_mma.cuh` · `src\ops\sparse_moe\prefill\sparse_moe_prefill_kernels.cu` |
| 9 | **启动参数与校验** | `src\serve\serve_options.cpp` |

每条改动的**为什么**与**实测读数**在 `docs/方案/`：
- 环与复用的完整评估 → `docs/方案/KVMem-复用路线完整评估-20260930.md`

---

## 4. 诚实清单（别把噪声当改动）

1. **`src\runtime\engine\device_profiles.json.bak-20260930-094325`** —— 这是标定档位时**误留在树里的备份文件**（44 件新增里的那 1 份）。它**不是改动**，**没有**收进 `changed-files\`；建议你在自己的树上也删掉。
2. **`REMOVED = 0`** —— 本树不缺上游任何文件；`__pycache__\*.pyc` 是运行期缓存，两边比对时都排除。
3. **`changed-files\` 是"改后的完整文件"，不是逐行 diff** —— 想看"改了哪几行"，用 §2 那条 `git diff --no-index` 命令自己生成；逐行 dump 没有随本仓发布。
4. **本仓不含构建产物**：不提供 `.exe` / `.dll` / `.lib` / `.obj`，也不含任何 `.ninfer` 制品或权重。
5. 许可与署名义务见仓根 `NOTICE.md`（Apache-2.0 §4）。
