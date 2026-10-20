<#
.SYNOPSIS
  R-2/R-3 回归夹具：ring + --auto-long-anchors 在「池 534 页 < 逻辑 262,144 token」下的超池 429 复现/验收。

.DESCRIPTION
  序列：3 段 → 4 段 → 5 段 → 同题重发 → 16.5K 短请求（每段一条 user message，制造消息边界让 auto-long-anchors 有落点；
  最后一问独立 user message，每轮重写 ⇒ 前缀分叉）。
  判据：
    R1' 3/4/5 段全 200
    R2' 日志无 500、无 `retained prefix exceeds its active KV entitlement`
    R3' 5 段之后紧跟的 16.5K 短请求必须 200（不重启引擎）
    R4' 负控：不传 `--auto-long-anchors` 时同夹具全 200（由 -RunNoAnchorNegControl 打开）

  负控（本脚本的核心防假绿）：
    -Expect unfixed  ⇒ 要求「未修版本必然红」：3/4/5 中至少一段非 200 且短请求非 200，并打印 RED-EXPECTED；
                         若全 200 则打印 FALSE-GREEN 并以 exit 2 失败。
    -Expect fixed    ⇒ 要求全绿（R1' + R2' + R3'，打开 -RunNoAnchorNegControl 时连 R4'）；
                         否则打印失败点并以 exit 1 失败。

.PARAMETER Expect
  unfixed（默认，先证明未修版本能复现红）| fixed（修后验收）

.PARAMETER RunNoAnchorNegControl
  再起一次不带 --auto-long-anchors 的引擎跑同一夹具，验收 R4'

.NOTES
  - 只读 + 起引擎；不编译、不改源码。运行需要用户单独授权 GPU。
  - 所有路径都是绝对路径；参数集中在 param() 块。
  - 引擎 exe 名是 `ninfer-serve.exe`（带 n）。
#>
[CmdletBinding()]
param(
    [ValidateSet('unfixed', 'fixed')][string]$Expect = 'unfixed',
    [switch]$RunNoAnchorNegControl,

    [string]$Exe = 'E:\infer-build\fusion-mine2\build-89\apps\ninfer-serve.exe',
    [string]$Model = 'E:\Ternary-Bonsai-2-27B-ninfer-v3.ninfer',
    [string]$Filler = 'E:\infer-build\kv-stream\long-prompt-27b.txt',
    [string]$WorkRoot = 'E:\infer-build\push-main-20261019\verify\读数-20261020-ring-anchor',

    [int]$Port = 8091,
    [int]$MaxContext = 262144,
    [int]$PoolTokens = 34176,   # = 534 页 x 64
    [int]$SegChars = 66000,     # 每段约 16.5K token（ASCII 4 chars/token）
    [int]$MaxTokens = 32,
    [int]$ReadyTimeoutSec = 180,
    [string[]]$EngineEnv = @('NINFER_KV_WINDOW=32768', 'NINFER_HOST_PAGEABLE=1', 'NINFER_KV_REUSE_HOSTBACKED=1')
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null

# ---- 环境（仅本进程；不改系统）------------------------------------------------
$VsVars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$CudaPaths = @('E:\cuda-13.3\bin\x64', 'E:\cuda-13.3\bin') | Where-Object { Test-Path -LiteralPath $_ }
$env:PATH = (($CudaPaths + $env:PATH) -join ';')
foreach ($pair in $EngineEnv) {
    $kv = $pair -split '=', 2
    Set-Item -Path "Env:$($kv[0])" -Value $kv[1]
}

# ---- 断言：一次只起一个引擎 / 端口空闲 ----------------------------------------
function Assert-NoResidualEngine {
    $procs = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'ninfer|infer-serve|llama-server' })
    if ($procs.Count -ne 0) {
        $procs | ForEach-Object { Write-Host ("  residual pid=" + $_.ProcessId + " name=" + $_.Name) }
        throw "residual engine process(es) present; stop them first (one engine at a time)"
    }
}
function Assert-PortFree {
    $listener = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -eq $Port })
    if ($listener.Count -ne 0) { throw "port $Port is already listening" }
}

# ---- 引擎起停 -----------------------------------------------------------------
function Start-Engine {
    param([bool]$WithAnchors, [string]$LogPath)
    Assert-NoResidualEngine
    Assert-PortFree
    $argv = @(
        $Model, '--host', '127.0.0.1', '--port', "$Port", '--model-id', 'bonsai2-27b',
        '--max-context', "$MaxContext", '--kv-capacity', "$PoolTokens", '--kv-dtype', 'rk8v4',
        '--host-kv-mib', '12288', '--max-concurrency', '1', '--max-shared-prefixes', '0',
        '--prefill-chunk', '256', '--gdn-state-fp16', '--no-cuda-graph'
    )
    if ($WithAnchors) { $argv += @('--auto-long-anchors', '--long-anchor-spacing', '1024') }
    if (-not (Test-Path -LiteralPath $Exe)) { throw "engine exe missing: $Exe" }
    if (-not (Test-Path -LiteralPath $Model)) { throw "model missing: $Model" }
    $proc = Start-Process -FilePath $Exe -ArgumentList $argv -NoNewWindow -PassThru `
        -RedirectStandardError $LogPath -RedirectStandardOutput ($LogPath -replace '\.err\.log$', '.out.log')
    $deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 1200
        if ($proc.HasExited) { break }
        if (Test-Path -LiteralPath $LogPath) {
            $text = Get-Content -LiteralPath $LogPath -Raw -ErrorAction SilentlyContinue
            if ($text -match 'listening on http://127\.0\.0\.1:') { return $proc }
        }
    }
    $tail = if (Test-Path -LiteralPath $LogPath) { Get-Content -LiteralPath $LogPath -Tail 20 } else { @() }
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    throw ("engine did not become ready in $ReadyTimeoutSec s; log tail:`n" + ($tail -join "`n"))
}
function Stop-Engine {
    param($Proc)
    if ($null -ne $Proc -and -not $Proc.HasExited) { Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    $left = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'ninfer|infer-serve' })
    if ($left.Count -ne 0) { $left | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
}

# ---- 夹具 ---------------------------------------------------------------------
function Get-FixtureSegments {
    if (-not (Test-Path -LiteralPath $Filler)) { throw "filler missing: $Filler" }
    $lines = @(Get-Content -LiteralPath $Filler -Encoding UTF8 | Where-Object { $_.Trim().Length -gt 0 })
    if ($lines.Count -eq 0) { throw "filler is empty: $Filler" }
    $segments = @()
    for ($s = 1; $s -le 5; $s++) {
        $parts = New-Object System.Collections.Generic.List[string]
        $len = 0
        $j = $s * 37
        while ($len -lt $SegChars) {
            $line = $lines[$j % $lines.Count]
            $parts.Add($line)
            $len += $line.Length + 1
            $j++
        }
        $segments += ($parts -join "`n")
    }
    return $segments
}

function Build-Messages {
    param([string[]]$Segments, [int]$SegmentCount)
    $messages = @()
    for ($i = 0; $i -lt $SegmentCount; $i++) {
        $messages += @{ role = 'user'; content = $Segments[$i] }
    }
    $messages += @{ role = 'user'; content = 'Question: Reply with the exact digit 7 and nothing else.' }
    return $messages
}

function Invoke-Chat {
    param([string]$Label, [array]$Messages)
    $body = @{
        model = 'bonsai2-27b'; messages = $Messages; max_tokens = $MaxTokens
        temperature = 0; stream = $false
    } | ConvertTo-Json -Depth 8
    $t0 = Get-Date
    $status = -1
    $contentLen = 0
    $err = $null
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
            -ContentType 'application/json' -Body $body -TimeoutSec 600 -UseBasicParsing
        $status = [int]$resp.StatusCode
        try {
            $json = $resp.Content | ConvertFrom-Json
            if ($json.choices -and $json.choices[0].message.content) { $contentLen = ([string]$json.choices[0].message.content).Length }
        } catch { }
    } catch {
        if ($_.Exception.Response) {
            $status = [int]$_.Exception.Response.StatusCode
            try {
                $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
                $err = $reader.ReadToEnd()
                $reader.Close()
            } catch { $err = $_.Exception.Message }
        } else { $err = $_.Exception.Message }
    }
    $ms = [int]((Get-Date) - $t0).TotalMilliseconds
    $line = "[{0}] status={1} ok={2} ms={3} content_len={4} err={5}" -f `
        $Label, $status, ($status -eq 200), $ms, $contentLen, (($err -replace '\s+', ' ').Trim())
    Write-Host $line
    return [pscustomobject]@{ Label = $Label; Status = $status; Ms = $ms; Ok = ($status -eq 200); Error = $err }
}

function Invoke-EngineLogMatches {
    param([string]$LogPath, [string]$Pattern)
    if (-not (Test-Path -LiteralPath $LogPath)) { return @() }
    return @(Get-Content -LiteralPath $LogPath | Select-String -Pattern $Pattern | ForEach-Object { $_.Line })
}

function Run-Arm {
    param([bool]$WithAnchors, [string]$Name)
    $log = Join-Path $WorkRoot "engine-$Name-$(Get-Date -Format yyyyMMdd-HHmmss).err.log"
    $segments = Get-FixtureSegments
    Write-Host "== arm=$Name withAnchors=$WithAnchors log=$log"
    $proc = Start-Engine -WithAnchors $WithAnchors -LogPath $log
    $results = @()
    try {
        foreach ($k in 3, 4, 5) { $results += Invoke-Chat -Label "seg-$k" -Messages (Build-Messages -Segments $segments -SegmentCount $k) }
        $results += Invoke-Chat -Label 'resend-5' -Messages (Build-Messages -Segments $segments -SegmentCount 5)
        $short = Invoke-Chat -Label 'short-16.5k' -Messages (Build-Messages -Segments $segments -SegmentCount 1)
        $results += $short
        if (-not $short.Ok) {
            # Recovery probe (lead 2026-10-21): distinguish "single request failed, engine
            # self-heals" from "engine is dead". Second identical short request, recorded too.
            $retry = Invoke-Chat -Label 'short-16.5k-retry' -Messages (Build-Messages -Segments $segments -SegmentCount 1)
            $results += $retry
            Write-Host ("[short-retry] first={0} retry={1}" -f $short.Status, $retry.Status)
        }
    } finally {
        Stop-Engine -Proc $proc
    }
    return [pscustomobject]@{ Name = $Name; Log = $log; Results = $results }
}

function Show-Log-Highlights {
    param([string]$LogPath)
    $patterns = 'anchor-drop|adopt-gate|adopt declined|retained prefix|no free page for host-restore|no placement|capacity \||context cache \|'
    $lines = Invoke-EngineLogMatches -LogPath $LogPath -Pattern $patterns
    Write-Host '--- engine log highlights ---'
    if ($lines.Count -eq 0) { Write-Host '  (none)' } else { $lines | Select-Object -Last 20 | ForEach-Object { Write-Host ("  " + $_) } }
}

# ---- 主流程 -------------------------------------------------------------------
$anchors = Run-Arm -WithAnchors $true -Name 'anchors'
Show-Log-Highlights -LogPath $anchors.Log
$r = @($anchors.Results)
$r1 = @($r | Where-Object { $_.Label -match '^seg-[345]$' -and -not $_.Ok })
$short = @($r | Where-Object { $_.Label -eq 'short-16.5k' })[0]
$r1Pass = ($r1.Count -eq 0)
$r3Pass = ($short -and $short.Ok)
$bad = Invoke-EngineLogMatches -LogPath $anchors.Log -Pattern 'retained prefix exceeds its active KV entitlement|HTTP 500'
$r2Pass = ($bad.Count -eq 0)

Write-Host ("R1' three/four/five segments all 200: {0}" -f $r1Pass)
Write-Host ("R2' no retained-prefix/500: {0}" -f $r2Pass)
Write-Host ("R3' short 16.5K after 5-seg is 200 without restart: {0}" -f $r3Pass)

$r4Pass = $null
if ($RunNoAnchorNegControl) {
    $neg = Run-Arm -WithAnchors $false -Name 'no-anchor'
    Show-Log-Highlights -LogPath $neg.Log
    $r4Pass = @($neg.Results | Where-Object { -not $_.Ok }).Count -eq 0
    Write-Host ("R4' no-anchor negative control all 200: {0}" -f $r4Pass)
}

# ---- 判定（负控防假绿）--------------------------------------------------------
if ($Expect -eq 'unfixed') {
    if ((-not $r1Pass) -and (-not $r3Pass)) {
        Write-Host 'RED-EXPECTED: unfixed baseline reproduced (R1 and/or R3 non-200). This is the proof the fixture discriminates.'
        exit 0
    }
    Write-Host 'FALSE-GREEN: unfixed expectation was RED, but fixture came back all-200. Check engine binary/params; do not trust this script.'
    exit 2
}

$allGreen = $r1Pass -and $r2Pass -and $r3Pass -and ($null -eq $r4Pass -or $r4Pass)
if ($allGreen) {
    Write-Host 'GREEN: R1/R2/R3 (and R4 when run) pass on this binary.'
    exit 0
}
Write-Host ("RED: fixed expectation not met. R1={0} R2={1} R3={2} R4={3}" -f $r1Pass, $r2Pass, $r3Pass, $r4Pass)
exit 1
