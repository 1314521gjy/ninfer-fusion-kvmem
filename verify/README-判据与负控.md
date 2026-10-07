# `verify/` · 判据脚本（**每个都写了"怎么变红"**）

> 这一组脚本是**交付件里实际跑过的那几个**：它们在"分发包"上跑出过 `KITCHECK_VERDICT=PASS (0 failures)`。
> 之所以连脚本一起发，是因为**白皮书里的结论都要能被别人重跑**——只会变绿的判据等于没有判据。

## 0. 运行前提

- Windows + **Windows PowerShell 5.1**（脚本是纯 ASCII，故意如此：5.1 会把无 BOM 的 `.ps1` 按 ANSI 读，
  脚本里写中文就会静默出错）；中文文件名一律**按内容特征查找**，不在脚本里写字面量。
- 需要有一个待校验的目录（"包"）。`-Kit <目录>` 指到它；不给就取脚本所在目录的上一层。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify\verify-kit-manifest.ps1 -Kit <包目录>
```

## 1. 五个脚本各管什么

| 脚本 | 它证明什么 | 判据行（成功时） | **负控（怎么让它变红）** |
|---|---|---|---|
| `verify-kit-manifest.ps1` | 清单里每个文件的 sha256 与磁盘一致 | `ok=<N> mismatch=0 missing=0` | 改一个字节 ⇒ `mismatch`；删一个文件 ⇒ `missing` |
| `gate-内测包-清单与形状.ps1` | **双向**：清单里的都在 + 磁盘上没有"清单外"的文件；并检查引擎/DLL/启动器形状与启动器 argv 必需项 | `KITCHECK_VERDICT=PASS` | ① 新增一个未入册文件 ⇒ `unlisted`；② 抽掉一个 DLL ⇒ `runtime dll` 变红；③ 把启动器里的 `--max-shared-prefixes 0` 删掉 ⇒ 该行变红 |
| `gate-ptq1档-清单与形状.ps1` | 同上，针对 PTQ1 档（含 `NINFER_TERNARY_PTQ1_FAST=1` 这条负载开关） | `KITCHECK_VERDICT=PASS (0 failure(s))` | 删掉启动器里那行 `NINFER_TERNARY_PTQ1_FAST=1` ⇒ 变红 |
| `自检-引擎与卡匹配.ps1` | 本机这张卡能不能跑这个包（读 `nvidia-smi` 的 compute capability 与引擎支持的架构比对） | `HWCARD_VERDICT=PASS` | 拿 `sm_86` 的包在 `sm_89` 的卡上跑 ⇒ 架构不符，明确报错而不是"慢" |
| `verify-arch-engine.ps1` | **本卡端到端验收**：起服务 + 过池告警臂（题面**中段**针）+ 装得下臂 | 起服务成功、过池臂答对中段针、装得下臂无告警 | 题面**中段针丢失 = FAIL**；`finish_reason=length` 或空答案 ⇒ **INCONCLUSIVE（不算漏针）** |
| `verify-tests-ctest-negcontrol.ps1` | 受影响测试真跑通，且失败分类表被测试钉住（**源码批 2026-10-07 实测 `VERDICT subset = PASS`**） | `VERDICT subset = PASS` + `negative control verdict: PASS(...)` | 翻 `failure_class.h` 一行 ⇒ ctest 必须 **failed** 并打出 `std::bad_alloc must classify as Capacity`；字节级还原后必须回到 **passed** |
| `verify-b01-overpool-429.ps1` | 装不下的请求被**可见拒绝**且引擎不被打死（429 + 编号消息 + 后续 200 + 存活 + 指标） | `refused with 429 = True` · `message carries a number = True` · `engine still serving (200) = True` · `server survived = True` | `-ShortProbeOnly`（不打断言）⇒ `refused` 必须 **False**、指标必须 **0**；`-ExpectStartRefusal` 是守卫正控，⚠️ **在本仓二进制上不成立**（见 B21），别拿它当红控 |
| `verify-256k-nonring-regression.ps1` | 关环（KVMem off）下的 256k 长题 + 4 路并发 + checkpoint 复用，用的是 PR #2 自带的校验脚本 | `script exit = 0` 且报告每条 `passed=True`；`near_256k.prompt_tokens ∈ [250000, 262144]`；`cached_tokens > 0` | `-RedControl`（`--long-rows 200`）⇒ `script exit` 必须非 0（实测 **1**） |
| `resolve-build-env.ps1` | **不是判据脚本**：上面两个需要工具链的脚本用它从**被测构建目录的 `CMakeCache.txt`** 反推 cl/ninja/CUDA 根/vcpkg 根，vcvars 用 `vswhere` 找 —— 所以这些脚本不带作者的机器路径 | `Show-BuildEnv` 打印出的每项都非空 | 任一字段取不到 ⇒ 脚本**打印** `could not derive <字段>`（不静默兜底）；把 `-BuildDir` 指到一个没配过的目录 ⇒ 直接以 "no CMakeCache.txt" 报错退出 |

## 2. 两条使用纪律（踩过才写的）

1. **`/v1/models` 返回 200 不等于能服务** —— worker 死后它照样 200。判活必须**真发一条请求**。
2. **同一个探针别用两种读法**：本篇所有"红控"都要在**同一支二进制**上做（例如检索窗口 64 → 256），
   否则你分不清是改动生效了还是换了个东西。
3. **改头不触发重编（本树实测 2026-10-07）**：本构建用 CMake 的 scanned C++20 规则，`DEP_FILE` 指 `<obj>.ddi.d`
   （**模块**依赖）⇒ **普通 `#include` 的头不是该 obj 的输入**，改它之后 `ninja` 会回答 `no work to do`。
   照原样读，你会拿**旧二进制**的绿灯当成新代码的结果（我们第一次跑分类表负控就是这样得了假绿）。
   修法：负控里**显式删掉 `.obj` 与 exe** 强制重编（`verify-tests-ctest-negcontrol.ps1` 已这么做并打印 `forced: …`）。
   系统含义：**这棵树的增量构建对头文件改动不可靠**，改头后请整目标重编。
4. **未知开关会被静默忽略**：参数解析链以 `src/serve/serve_options.cpp:992`
   `} else if (parse_dispatch_options(arg)) { }` 结束，**没有 else-throw**，全文件只有 `--help` 与 `argv[1]` 两处校验。
   ⇒ 每个开关都要对源码核实，并抓服务器**自报字段**反证它生效了（`INFO capacity | KV <N> tokens, <dtype>, explicit | pages X/Y`）；
   本机就踩过：脚本里传的 `--no-kv-lease-growth` **在本树不存在**，被静默吞掉，而那一跑看起来是"成功"的。
   ⭕ **2026-10-07 复核：本条的前提不成立（实测证伪）** —— 引擎**会拒绝**未知开关：那个
   `parse_dispatch_options` lambda **自己**以 `else { throw std::invalid_argument("unknown argument: " + arg); }`
   结束（`serve_options.cpp:709-711`）。四臂实测（本仓件与出厂件）：`--definitely-not-a-flag`、
   `--no-kv-lease-growth`、`--vram-reserve-mib` **全部** `unknown argument: X` + **exit 1** + usage 全文；
   对照臂「不给开关」引擎继续往下走 ⇒ 两态可区分。**同一事实在 `docs/02-排错手册`（§1 表）
   与 `docs/08-从零复现兜底` 里记的就是正确行为**（`unknown argument: --disk-cache` ⇒ 删掉它）。
   **本条里仍然成立的**：① 要抓服务器自报字段反证配置生效（这永远成立）；② 别的批（F2 之后）的启动器
   参数**照抄到本树会启动即拒**（`--no-kv-lease-growth` 只有肯定式的 `--kv-lease-growth`，默认 `false`）。
5. **"真红被读成绿"：崩溃 / 缺 DLL 的用例会被漏数（2026-10-07 实测，已修）** —— `verify-tests-ctest-negcontrol.ps1`
   原先有三处叠加缺陷：① 用 `[xml](Get-Content -Raw)` 读 JUnit（前言声明 `encoding="UTF-8"` 时 .NET 抛异常）⇒ 返回 `$null`；
   ② 按 `<testsuites><testsuite>` 找用例，而本项目 JUnit 的**根元素就是 `<testsuite>`** ⇒ **永远解析到 0 条**；
   ③ 退回控制台解析的正则只认 `Passed|Failed|Skipped|Timeout`，而崩掉的用例打的是
   `… Exit code 0xc0000135***Exception: 550.57 sec` ⇒ **该用例被直接丢掉**。
   **后果（当真发生过）**：4 个用例里 3 个崩，脚本报 `passed=1 failed=0 of 1` ⇒ **`VERDICT = PASS`**，而 `ctest` 自己返回 8。
   **修法**：JUnit 改成按路径 `XmlDocument.Load()` 读、两种根元素都接受、控制台正则补崩溃形态，并加两条**硬不变量** ——
   **`ctest` 自己的返回码**与**子集应有条数（`ctest -N -R` 的 `Total Tests`）**必须与解析结果一致，否则一律 `NOT-CLEAN` 并打印原因。
   **规矩**：解析器的"少算/漏报"必须由**外部权威**兜住，不能只靠 `tests>0 且 failed==0` 这种自证式判据。
6. **跑 ctest 前把 CUDA bin 放进 PATH**（`E:\cuda-13.3\bin\x64`；`cudart64_13.dll` 只在那里，`bin\` 下没有）：
   否则多数用例以 **`0xC0000135`（STATUS_DLL_NOT_FOUND）** 失败，**而且可能表现为"先卡几分钟再失败"**（实测 `ninfer_hadamard_transform_test` 崩前挂了 550 s）。
   **这不是代码回归** —— 同一二进制带上 `bin\x64` 后 4/4 通过（2.31 s）。

## 3. 与白皮书的关系

白皮书 §8.4 的判据表逐条对应本目录的脚本；§9 的读数就是这些脚本与孪生脚本（模拟安装、门禁跑台）跑出来的。
读数口径、正控/负控/红控的分类见白皮书 §9–§10。

## 4. 不在这里的东西

- **模型体检**（读 `.ninfer` 容器、判断带什么投机头、给出该用哪个 `--spec`）在 `tools/自检-模型件.ps1`。
- **模拟安装**（把包复制到全新 ASCII 路径、走完接收方全部步骤）与**批量重算清单/门禁**的脚本是本机运维脚本，
  带硬编码路径，未随仓发布；白皮书 §9.3 的读数是它们跑出来的，方法在 §8 里逐字写清了。
