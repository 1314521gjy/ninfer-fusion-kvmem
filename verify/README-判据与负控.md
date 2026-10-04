# `verify/` · 判据脚本（**每个都写了"怎么变红"**）

> 这一组脚本是**交付件里实际跑过的那几个**：它们在"内测包"上跑出过 `KITCHECK_VERDICT=PASS (0 failures)`。
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

## 2. 两条使用纪律（踩过才写的）

1. **`/v1/models` 返回 200 不等于能服务** —— worker 死后它照样 200。判活必须**真发一条请求**。
2. **同一个探针别用两种读法**：本篇所有"红控"都要在**同一支二进制**上做（例如检索窗口 64 → 256），
   否则你分不清是改动生效了还是换了个东西。

## 3. 与白皮书的关系

白皮书 §8.4 的判据表逐条对应本目录的脚本；§9 的读数就是这些脚本与孪生脚本（模拟安装、门禁跑台）跑出来的。
读数口径、正控/负控/红控的分类见白皮书 §9–§10。

## 4. 不在这里的东西

- **模型体检**（读 `.ninfer` 容器、判断带什么投机头、给出该用哪个 `--spec`）在 `tools/自检-模型件.ps1`。
- **模拟安装**（把包复制到全新 ASCII 路径、走完接收方全部步骤）与**批量重算清单/门禁**的脚本是本机运维脚本，
  带硬编码路径，未随仓发布；白皮书 §9.3 的读数是它们跑出来的，方法在 §8 里逐字写清了。
