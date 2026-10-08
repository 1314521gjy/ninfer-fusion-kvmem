# NInfer Fusion KVMem

**面向 Windows / NVIDIA GPU 的长上下文推理实验：用 KVMem 环、主机 KV 存储与按需检索，让设备 KV 池可以小于逻辑上下文，并在后续轮次复用已有上下文。**

本项目基于 NInfer `0.11.0-rtx3090`，发布融合后的源码、复现记录和验证工具。它是 NInfer 的**下游衍生项目**；基座与第三方算法来源见 [署名与改动声明](NOTICE.md)。

> [!WARNING]
> 这是初期实验版本，仍有未定位问题，不保证回答正确性与运行稳定性。维护者建议用于个人探索，不建议用于商用；该建议不改变 [Apache-2.0 许可](LICENSE)。使用前请读 [Bug 账本及状态复核](已知问题-初期版本.md)。

[获取引擎](https://github.com/1314521gjy/ninfer-fusion-kvmem/releases) · [编译指南](编译指南-怎么编.md) · [复现白皮书](复现白皮书-NInfer-KVMem环-20261002.md) · [实测回执](实测回执与反馈.md) · [问题反馈](https://github.com/1314521gjy/ninfer-fusion-kvmem/issues)

## 这个项目做了什么

长上下文会增加 KV 缓存占用。本项目将三个容量分别配置：

| 配置 | 含义 | 示例 |
|---|---|---|
| `--max-context` | 单序列的逻辑上下文上限 | 262,144 token |
| `--kv-capacity` | 设备端 KV 池容量 | 17,920 token（280 页） |
| `--host-kv-mib` | 主机端 KV 存储预算 | 16,384 MiB |

KVMem 环保留常驻窗口，将其他页降到主机层；需要时按内容相关性选页、搬回设备并装配注意力窗口。host-backed 复用与惰性物化让后续轮次可以继续使用主机驻留的检查点。按卡档位则为内核调度选择路由。

**小 KV 池不代表模型权重也能装进小显存**，也不代表检索与完整注意力在任意任务上质量等价。模型权重、运行工作区、投机头、CUDA Graph 和主机内存都需要单独预算。机制、容量公式与负控见 [白皮书](复现白皮书-NInfer-KVMem环-20261002.md) §2–§4。

## 从哪里开始

| 你的目标 | 入口 |
|---|---|
| 先试运行服务 | 下方“获取与启动”；先准备引擎、依赖和模型 |
| 自己编译或修改引擎 | [编译指南](编译指南-怎么编.md)与 [完整源码树](src-tree/fusion-engine-src/) |
| 重跑长上下文与复用实验 | [复现白皮书](复现白皮书-NInfer-KVMem环-20261002.md) §8–§10、[验证脚本说明](verify/README-判据与负控.md) |
| 看实测结果和反例 | [实测回执](实测回执与反馈.md)、[Bug 账本](已知问题-初期版本.md) |
| 核对项目贡献与第三方来源 | [新增能力与已解决问题](我方-新增能力与已解决问题.md)、[NOTICE](NOTICE.md) |
| **最短验收清单（照用）** | 下方“怎么确认 KVMem 与复用在工作”一节，加 [验证脚本说明](verify/README-判据与负控.md) —— 每条判据都写了**“怎么变红”**，只会变绿的判据不算判据 |

## 获取与启动

### 1. 准备引擎、依赖与模型

源码可以直接克隆；运行引擎和模型时，使用纯 ASCII 路径，例如 `D:\ninfer\`，避免已记录的中文路径读取问题。

```powershell
git clone https://github.com/1314521gjy/ninfer-fusion-kvmem.git
cd ninfer-fusion-kvmem
```

引擎有两种获取方式：从 [Releases](https://github.com/1314521gjy/ninfer-fusion-kvmem/releases) 下载，或按下方步骤编译当前源码。

截至 **2026-10-07**，公开 Release [`engine-v0.11.0-kvmem-20261003`](https://github.com/1314521gjy/ninfer-fusion-kvmem/releases/tag/engine-v0.11.0-kvmem-20261003) 提供：

| 文件 | 目标架构 | 对应显卡系列 |
|---|---|---|
| `ninfer-serve-86.exe` | `sm_86` | RTX 30 系 |
| `ninfer-serve-89.exe` | `sm_89` | RTX 40 系 |
| `ninfer-serve-120a.exe` | `sm_120a` | RTX 50 系 |
| `SHA256SUMS.txt` | — | 上述引擎的哈希清单 |

这些资产**不是完整运行套件**：该 Release 没有附带 DLL 或模型。请按对应构建的依赖准备 CUDA / FFmpeg / libcurl 运行库，将所需 DLL 放在 EXE 同目录；具体依赖与已记录的分发待办见 [NOTICE](NOTICE.md) §6–§7。可用 `Get-FileHash <引擎路径> -Algorithm SHA256` 与下载的清单对照。

**源码与 Release 要分别看**：`main` 上 2026-10-07 的修复与验证记录，不能作为 2026-10-03 引擎资产已包含这些修复的证明。测试时请记录源码提交或 EXE 哈希。

本仓库与该 Release **均不含模型权重或 `.ninfer` 模型制品**。需要另行准备兼容的 NInfer v3 模型；转换说明见 [源码树的权重转换文档](src-tree/fusion-engine-src/docs/weight-conversion.md)。从仓库根目录先做模型体检：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\自检-模型件.ps1 `
  -Model D:\ninfer\models\model.ninfer -VramGb 16
```

将路径与 `16` 替换为实际模型和显存容量。体检只读文件，不启动引擎；裸件输出 `MODELCHECK_VERDICT=BARE`、退出码 3，表示可读但没有投机头。损坏或截断的文件应先重新获取。

### 2. 启动一个文本服务

下面是 **sm_89 / RTX 40 系的参数模板**，不是所有显卡的通用配方。将引擎与模型路径替换为本机实际路径；自编译时引擎文件名为 `ninfer-serve.exe`。该示例分配 16 GiB 主机 KV 预算，需另留模型及运行工作区的显存、主机内存。

在 PowerShell 中设置环境变量并启动，变量必须由启动引擎的同一个终端传入：

```powershell
$engine = 'D:\ninfer\engine\ninfer-serve-89.exe'
$model = 'D:\ninfer\models\model.ninfer'

$env:NINFER_KV_WINDOW = '16384'
$env:NINFER_KV_RETRIEVE = '8192'
$env:NINFER_KV_RING = '1'
$env:NINFER_HOST_PAGEABLE = '1'
$env:NINFER_KV_REUSE_HOSTBACKED = '1'
$env:NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL = '64'

& $engine $model `
  --host 127.0.0.1 --port 8091 --model-id ninfer-local `
  --max-context 262144 --kv-capacity 17920 --kv-dtype k8v4 `
  --host-kv-mib 16384 --prefill-chunk 1024 `
  --max-concurrency 1 --max-shared-prefixes 0 `
  --default-max-tokens 1024 --default-reasoning-effort none
```

模板不启用投机解码或视觉，以免假定模型含有相应组件。需要投机解码时，以模型体检输出为准：

| 模型组件 | 可用参数 |
|---|---|
| 仅 `text` | 不加 `--spec` |
| 含 `mtp` | `--spec mtp`；MTP-only 件不要加 `--lm-head-draft` |
| 含 `dflash2` 与对应 proposal head | 按模型配方使用 `--spec dflash2 --draft-tokens 4 --lm-head-draft` |

小卡还需核对权重与工作区占用，不能直接照搬大卡的 dflash2 配方。CUDA Graph 捕获失败时，可尝试 `--no-cuda-graph`，这是已记录的绕道，根因仍未定位。

### 3. 发一条真实请求

在另一个 PowerShell 终端执行：

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8091/v1/models'

$body = @{
  model = 'ninfer-local'
  messages = @(@{ role = 'user'; content = 'What is 17 + 25? Reply with the number only.' })
  max_tokens = 64
  stream = $false
} | ConvertTo-Json -Depth 6

$response = Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8091/v1/chat/completions' `
  -Method Post -ContentType 'application/json' -Body $body
$response.choices[0].message.content
```

预期答案是 `42`。`/v1/models` 返回 200 只证明模型列表接口可访问；必须检查真实生成请求、答案与进程存活，才能判断服务是否可用。其他 API 与响应字段见 [服务文档](src-tree/fusion-engine-src/docs/serving.md)。

### 4. 控制思考（reasoning）通道（可选）

两个开关都**只在开启思考时生效**：

| 想要 | 怎么做 |
|---|---|
| 限制单轮思考长度 | 请求体加 `"thinking_budget": 1500`（OpenAI 路由，与 Anthropic 路由同义）。`0` / 省略 / `null` = 不限；**显式给小值（< 1024）会被拒**：HTTP 400 并点名 `thinking_budget` |
| 抑制思考里的逐字复读 | 启动加 `--thinking-presence-penalty 2.0`（范围 −2..2）。**只作用于思考通道** |
| **客户端不带 `thinking_budget`，但不想看到空回复** | **服务端加 `--default-thinking-budget 1024`**（服务端默认预算）。只在该请求**没带**该字段时生效；请求里显式给值仍以**请求**为准；`0` 或越 uint32 会被拒。四臂读数见 [verify/读数-20261008/](verify/读数-20261008/) |
| 读请求日志要认版本号 | `schema_version` 现为 **28**：`result` 段新增 5 个复读遥测字段 `repeat_channel` / `repeat_tokens` / `repeat_uniq8` / `repeat_dup8` / `repeat_max8`（纯新增，无字段改义） |

实测（本机 RTX 4080 SUPER、`--greedy`）：预算 1500 把 `reasoning_tokens` 从 2699 压到 1525、用时 52.7 s → 29.9 s；惩罚 +2.0 把思考的 8-gram 复读率从 0.2695 降到 0.1922（等长前缀口径），而**正文通道逐字不变**。读数、口径与"怎么变红"见 [实测回执与反馈.md](实测回执与反馈.md) §9。

## 怎么确认 KVMem 与复用在工作

短问答成功后，再做长上下文测试；“服务能回答”与“超池检索正确”是两项验收。

1. **核对启动配置**：日志应出现 `engine ready`、实际 `capacity | KV … | pages …` 与 `reuse host-backed: on`。默认启用内容打分时还会出现 `[ring] content scoring ON by default`；检查是否被旧环境变量显式关闭。
2. **核对检索路径**：超窗题面应能看到 `kvmem_score: SELECT … query_tokens=…`。题面短于窗口时没有 SELECT 属正常情况；单有日志不能证明答案正确。
3. **核对答案与第二轮**：在长题面的中段放入可验证的内容，提问并追问，检查答案、缓存复用字段和 TTFT。空答案或 `finish_reason=length` 要单独记录，不能直接记作检索漏针。
4. **做同一二进制上的对照**：按白皮书的固定夹具，将 `QUERY_TAIL` 从 64 改为 256，或关闭检索，验证判据能区分正控与负控。历史掉针比例只适用于相应夹具。

脚本入口见 [verify/README-判据与负控.md](verify/README-判据与负控.md)。包清单脚本需要实际分发包和清单；模型体检、包哈希检查、真实请求、检索质量分别验证不同事项，不能互相替代。

## 从源码编译

优先编译 [当前完整源码树](src-tree/fusion-engine-src/)。`patches/changed-files/` 是早期一批改后的完整文件，**不是全部当前改动或逐行 diff**；不应只覆盖它就声称得到当前 `main`。

Windows 构建需要 NVIDIA 驱动、CUDA Toolkit、MSVC x64 工具链、CMake、Ninja 与 vcpkg。仓内构建指南记录的组合是 **CUDA 13.3 / VS 18 BuildTools**；当前 `CMakeLists.txt` 要求 **CMake ≥ 3.28**，`vcpkg.json` 固定依赖基线。其他工具链的兼容性需自行验证。

在已加载 MSVC x64 环境的 **cmd.exe** 中，将路径替换为实际安装目录，再执行：

```bat
set "VCPKG_ROOT=D:\vcpkg"
set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"

cd /d D:\ninfer-fusion-kvmem\src-tree\fusion-engine-src
cmake -S . -B build-89-clean -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCMAKE_CUDA_COMPILER="%CUDACXX%" ^
  -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF

cmake --build build-89-clean --target ninfer-serve -j 8
```

产物位于 `build-89-clean\apps\ninfer-serve.exe`。RTX 30 / 50 系分别使用 `86` / `120a`，并各用一个新的构建目录。CMake 还允许 `80`，源码将其标为未实测兼容目标；允许配置不等于已验。

构建注意事项：

- **每个架构使用新的空构建目录**。混用不同时期的对象文件曾导致启动崩溃，见 [0x18c729 根因与修法](根因与修法-0x18c729-20261003.md)。
- **改头文件后确认相关目标确实重编**。仓内 2026-10-07 记录过增量构建漏掉头文件改动的情况；使用新目录可避免拿旧二进制验新代码。
- **编译不需要模型**；首次 vcpkg 依赖构建需要网络，CUDA 编译需预留内存。完整环境、驱动要求和 FFMPEG 排错见 [编译指南](编译指南-怎么编.md)。

## 已记录的实测结果

以下是维护者记录的特定机器、模型制品和夹具结果，**不是通用 benchmark 或复现保证**。完整日期、日志名和条件见 [实测回执](实测回执与反馈.md)与 [白皮书](复现白皮书-NInfer-KVMem环-20261002.md) §9。

> ⚠️ **未验纪律（本仓的读法，请照此读上面和下面每一张表）**：任何读数**只在写明的日期、卡型、模型件与配置上成立** ——「没测的明写未验，**不许当已验读**」。要把某一格推广到别的卡、别的模型件或别的上下文长度，**先按白皮书 §8 重跑一遍**再回来读这张表；**反例与阴性结果同样保留**（例如上表“约 95k 零命中”一行，以及 `已知问题` 里标着「未复现／未定位」的条目）。

| 场景 | 已记录读数 | 条件与边界 |
|---|---|---|
| 小池多针题面，两轮问答 | 6/6 + 6/6；`TAIL=256` 对照第二轮 2/6 | 2026-10-02；申请 4,000 token 池，按 64 token/页取整为 4,032 token；低池验收用 `k8v4` |
| 94,698 token 长题 | 3/3 + 3/3 | 显式 `QUERY_TAIL=64` 且确认打分运行的实验臂；另有约 95k 题面零命中反例，仍未定位 |
| 同会话长前缀复用 | 第二轮 cache 80,075，TTFT 70,829 ms → 126.5 ms | 白皮书 §4 的配置与题面；不代表任意请求都能达到该延迟 |
| PQ2 + dflash2，draft 12 | 解码 571.9 tok/s | RTX 4080 SUPER 本机记录、计数语料；采用 `req#1 done` 口径，不是一般任务速度 |
| RTX 5080 社区回执 | 104,991 token 题面中段针命中 | 2026-10-03，设备池为完整 262,144 token；不能当成该卡的小池检索验收 |
| 256k 非环回归 | 256,944 prompt token；4 路并发两轮通过 | 2026-10-07 源码批，KVMem 关闭；不能作为开环多并发的证据 |

主要本机记录来自 RTX 4080 SUPER / `sm_89`；另有 RTX 5080、3060、5070 Ti 与 5070 Ti Laptop 的社区回执。社区启动成功、单题命中与本机完整验收的范围不同，见 [回执原文](实测回执与反馈.md)。

## 当前边界与排错

> **8 GB 卡请用仓根的 `start-8g.bat`**（8 GB 专属；其他卡不要用）。它把 KV 精度降到 `rk4v4-e8`（每页 64 字 ≈1.10 MiB，bf16 是 4.0 MiB）、用 `NINFER_KV_WINDOW` 定常驻窗口、并让 `--kv-capacity auto` 按窗口定池 —— 实测本机整卡 6.84 GiB（权重 5.52 + 运行 1.32），窗口 65,536 字、逻辑上下文 131,072 字。质量代价按困惑度实测 **+0.30%**。读数与"怎么变红"见 `实测回执与反馈.md` §14。

[Bug 账本](已知问题-初期版本.md)保留历史状态，并附 **2026-10-07 状态复核与后续更正**。查状态时请读最新复核及对应证据，区分源码已修改、实际构建已验证和旧发布件行为。

| 现象或限制 | 处理与证据入口 |
|---|---|
| 开环或内容打分时需要多并发 | 当前实现只支持 `--max-concurrency 1`；源码启动守卫会拒绝不支持的组合（B06） |
| KVMem 环与 hybrid prefix cache 同开 | 不支持；环需要 Legacy Host KV 层，启动示例不使用 `--use-alt-prefix-caching` |
| 超池请求崩溃、容量拒绝或重发后不可用 | 保留 `--max-shared-prefixes 0`，客户端 `max_tokens` 不超过设备池 token 数；查看 B01 与对应构建回执，不照抄其他批次的开关 |
| 服务可访问但长题答错 | 检查检索配置并重跑固定夹具；B07 的约 95k 零命中反例仍未定位 |
| 关键内容跨页缝，只检索回一侧 | 已记录质量边界，不能据一次通过认定已解决 |
| 长时间冷预填后连接中断 | 连接层根因未定位；超时 / keep-alive 调整属于缓解 |
| 小显存卡装不下模型 | 核对权重、投机头、视觉和工作区预算；8 GB 配方仍需真机验证 |
| 8 GB 档用 `rk*` 系 KV dtype（`rk8v4`/`rk4v4`/`rk4v4-e8`/`rk2v4-e8`） | **超池 + 检索未开时不可靠**：失败形态是「**静默错**」（HTTP 200、答案貌似合理）而不是报错。2026-10-07 实测（同二进制、同夹具，**只改 `NINFER_KV_RETRIEVE`**）：超池臂 **检索关 ⇒ turn1 命中 0/6、静默错 6/6；检索开（8192）⇒ 6/6 命中**；不过池（题面装得下）两态都 2/2 ⇒ 那条塌是**检索可见性**问题。⇒ 小池 / 超池请**开着检索**，或把 `--kv-capacity` 抬到 ≥ 题面 token 数（记在 [Bug 账本](已知问题-初期版本.md) **B28**/B11） |
| 超池构型（设备池 < 逻辑上下文）要保住输出 | **不要开 `--kv-lease-growth`**（本仓默认**关**；F2 之后的批次默认开，照抄会命中"输出被砍到 2 个 token"那条回归）。⚠️ 反过来，`--no-kv-lease-growth` 这类**否定式开关在本仓不存在**，写了会**启动即拒**（见 B22） |
| **超池 + 检索**（任何 dtype） | 超池题的**对错取决于检索是否开着**：`rk4v4` 实测 **检索关 ⇒ 0/6 全静默错、检索开(8192) ⇒ 6/6 命中**（B28）。⇒ 超池时**别关检索**；并且读引擎那行 over-pool 告警（`prompt > pool`）——**它出现就说明该请求的答案不许当已核** |
| 跑 MTP + 想调并发（B10） | 显存预留与 `--max-concurrency` 互相挤：3090 上并发 4 时上下文上限 180,224 token，改成 1 后 262,144 可跑 ⇒ **先定并发，再定上下文** |
| 小显存卡用错模型件（B15） | 必须 `…-mtponly.ninfer` + `--spec mtp`；拿带视觉头/别的头的件去跑会崩或起不来 |
| 用 dflash2 当投机头（B18） | 其头质量明显低于 PQ2（接受率 22%/47% vs 91.5–100%），且**引擎不支持外挂 draft 模型** ⇒ 要高质量请用 PQ2 档的件 |
| 8 GB 档想直接启动（B09） | 配方已量到单点（三杠杆 `--gdn-state-fp16` + `--prefill-chunk 256` + `--no-cuda-graph` = **−289.9 MiB**，见上"超池"行），但**池上限是外推**（取决于真机 `free`）⇒ 本仓**已附** `start-8g.bat`（入仓 `b924cdf`，就是上面「8 GB 卡请用仓根的 `start-8g.bat`」那一行说的那个文件）；真机 `free` 与本机不同时，请按实测自校 `--kv-capacity` |
| 其它已知边界（B20） | `--draft-tokens 10` 无产物证据 · A 卡与 V100（sm_70）不支持 · 树内留有一个 `device_profiles.json.bak-*` · 一条「仅设 `NINFER_KV_RING` 才崩」的反馈**已核、属实、已修** —— 环的两道门读的不是同一个环境变量，照引擎自己的提示操作必崩，见 **B30** |

更多症状与绕道见 [排错手册](docs/02-排错手册%20·%20预案与处方.md)、[已知问题与禁忌](docs/05-已知问题与禁忌.md)和 [Bug 手册](docs/06-Bug手册.md)。部分 `docs/` 文档来自特定历史部署包，端口、模型名、容量和本机路径应按对应版本核对。

## 仓库结构与文档

```text
src-tree/fusion-engine-src/     当前融合引擎源码、内置依赖、测试与上游文档
patches/                       早期改动快照、说明与逐文件对账清单
docs/                          部署、调优、排错与原始研究记录
tools/                         模型容器检查、模型体检、冒烟与校验工具
verify/                        包校验、运行验证、回归与负控脚本
NOTICE.md / LICENSE            来源、改动声明与许可
```

- [复现白皮书](复现白皮书-NInfer-KVMem环-20261002.md)：机制、复现步骤、判据、负控与原始读数。
- [新增能力与已解决问题](我方-新增能力与已解决问题.md)：存储解耦、host-backed 复用、容量守卫、打分检索、量化内核与交付工具的贡献及上游对账。
- [部署操作手册](docs/12-给接收方-agent-的操作手册.md)：历史分发包的接入流程；其中的模型、启动器与套件文件不一定随本仓或 Release 提供。
- [文档索引](docs/README.md)、[补丁说明](patches/README-改动说明.md)、[验证脚本索引](verify/README-判据与负控.md)：按任务继续阅读。

`NOTICE.md` 记录的 **2026-10-03** 基线对账为：2,327 件相同、92 件修改、44 件新增、0 件缺失。这是有日期的历史快照；当前差异应以源码、提交和对账脚本重新核对。

## 反馈问题

请在 [Issues](https://github.com/1314521gjy/ninfer-fusion-kvmem/issues) 提供可复现信息：

- 源码提交或 Release 标签、引擎 SHA256；GPU / 显存 / 驱动 / Windows 版本。
- 模型来源、文件哈希、模型体检输出；完整启动参数与相关 `NINFER_*` 环境变量。
- 启动日志、请求参数、响应码、`finish_reason`、实际答案与预期答案。
- 长上下文问题补充 token 数、设备池 / 主机预算、是否第二轮、是否能用短题或负控区分。

请先去除日志中的密钥、私人提示内容与本机敏感路径。仅报告“端口在监听”或“模型列表返回 200”不足以复现生成失败。

## 来源、许可与引用

本仓内容按 [Apache License 2.0](LICENSE) 发布。各第三方目录保留自身许可证；模型权重按各自来源许可处理，本仓不分发权重。

- **引擎基座**：[Neroued/ninfer](https://github.com/Neroued/ninfer)、[ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090)与 [iamwavecut/ninfer-3090](https://github.com/iamwavecut/ninfer-3090)；本仓基线版本为 `0.11.0-rtx3090`。
- **KVMem 环来源**：[tancau/ninfer-kvmem-ring](https://github.com/tancau/ninfer-kvmem-ring)，含移植与共用改动面。
- **策略层来源**：[kvmem/kvmem-qw3](https://github.com/kvmem/kvmem-qw3)，作者 Di Chai；源码树包含其改造版及许可、notices。
- 逐文件来源、项目改动、CraneBW / laamaafung 的语义参照、第三方库与运行库分发待办，以 [NOTICE.md](NOTICE.md) 为准。

引用本项目时，请附使用的提交或 Release 标签，便于区分实验与构建版本：

```text
NInfer Fusion KVMem — KVMem 环、host-backed 复用与长上下文推理复现记录.
https://github.com/1314521gjy/ninfer-fusion-kvmem
Version: <commit SHA or release tag>; accessed: <YYYY-MM-DD>.
```
