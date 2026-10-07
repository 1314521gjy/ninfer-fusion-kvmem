# run-a-tier-batch.ps1 -- the "A tier" batch: command-level verifications in ONE pass.
#
# Reuse policy (do not reinvent): every section names the wheel it reuses. The only new code here is
#   (a) this driver loop, and
#   (b) Start-Arm -- a pooled start helper, because the existing probes hard-code their own argv/env
#       (probe-kmin-sweep.ps1 fixes the pool list + sink, probe-b06-guard.ps1 fixes the env set).
#
# SECTIONS
#   T2  three-lever combination effect            pool 64 pages, 5 argv variants, read runtime+free
#   T3  NINFER_KV_SINK two-copy consistency       SINK=640 (10 pages) -> [ring] budgets skeleton=?
#   T4  lean profile + --request-log-jsonl        pools 34/512 pages + one request each -> memory block
#   T5  kMin threshold, repeat (n was 1)          reuse probe-kmin-sweep.ps1, pools 58/59/60/61
#   T6  17920 delivered tier tool visibility       reuse probe-kmin-sweep.ps1, pool 280 pages
#   T7  --max-context exactly equal to the pool   pool 1024 pages, maxctx 65536 tokens -> ring on/off?
#   T1  FULL ctest suite (258)                    reuse run-tests.ps1, run LAST (it is the long one)
#
# NOTES THAT COST TIME BEFORE (kept here so they do not cost again)
#   * ctest children need E:\cuda-13.3\bin\x64 on PATH (cudart64_13.dll lives there) or they die with
#     0xC0000135, sometimes only after minutes.
#   * run-tests.ps1 prints via Write-Host, so `$out = & script` captures NOTHING -- read its artifacts
#     (<OutRoot>\test-run.tsv and ctest-subset.xml) instead.
#   * The pool line is the engine's own reading: "capacity | KV N tokens, DTYPE, explicit | pages X/Y |
#     runtime R | free F". The [ring] budgets line self-checks skeleton+evidence+recent+free == pool.
#
# Pure ASCII on purpose (Windows PowerShell 5.1 reads BOM-less files as ANSI).

param(
  [string]$Exe   = 'E:\infer-build\p0-20261003\_verify-b01\build-ninja\apps\ninfer-serve.exe',
  [string]$Model = 'E:\ship-next\modelpack-ptq1-20261002\bonsai2_27b_ternary_ptq1_native_mtp.ninfer',
  [string]$OutRoot = 'E:\infer-build\p0-20261003\_verify-b01\a-tier',
  [string]$TestsScript = 'E:\infer-build\p0-20261003\_verify-b01\run-tests.ps1',
  [string]$KminSweep = 'E:\infer-build\p0-20261003\probe-kmin-sweep.ps1',
  [int]$BasePort = 8600,
  [int]$ReadySec = 240,
  [switch]$SkipFullTests
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$script:Port = $BasePort
$rows = New-Object System.Collections.ArrayList
$tsv = Join-Path $OutRoot 'a-tier.tsv'

if (-not (Test-Path $Exe))   { Write-Host ("FATAL: exe not found: " + $Exe); exit 2 }
if (-not (Test-Path $Model)) { Write-Host ("FATAL: model not found: " + $Model); exit 2 }
$env:CUDA_PATH = 'E:\cuda-13.3'
$env:PATH = 'E:\cuda-13.3\bin;E:\cuda-13.3\bin\x64;' + (Split-Path $Exe -Parent) + ';' + $env:PATH

function Say([string]$t) { Write-Host ("[{0}] {1}" -f (Get-Date).ToString('HH:mm:ss'), $t) }

function Save-Rows {
  $rows | Export-Csv -Path $tsv -Delimiter "`t" -NoTypeInformation -Encoding UTF8
}

function To-Mib([string]$v, [string]$u) {
  $x = [double]$v
  switch ($u) { 'KiB' { return $x / 1024.0 } 'MiB' { return $x } 'GiB' { return $x * 1024.0 } default { return $x } }
}

# ---- the one new helper: start the engine with a caller-chosen pool/argv/env and read its self-reports
function Start-Arm {
  param(
    [string]$Section, [string]$Name,
    [int]$PoolPages = 64, [int]$MaxContext = 65536,
    [string[]]$Extra = @(), [string]$SinkTokens = '', [string]$Scorer = '',
    [switch]$SendRequest, [string]$RequestLog = ''
  )
  $port = $script:Port; $script:Port = $script:Port + 1
  foreach ($v in 'NINFER_KV_SINK','NINFER_KV_WINDOW','NINFER_KV_RING','NINFER_KV_RETRIEVE','NINFER_TERNARY_KVMEM',
                 'NINFER_TERNARY_KVMEM_SCORE','NINFER_HOST_PAGEABLE','NINFER_KV_REUSE_HOSTBACKED',
                 'NINFER_TERNARY_PTQ1_FAST','NINFER_KV_RETRIEVE_SHARE') {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
  }
  $env:NINFER_KV_RING = '1'; $env:NINFER_KV_WINDOW = '1536'; $env:NINFER_KV_RETRIEVE = '768'
  $env:NINFER_HOST_PAGEABLE = '1'; $env:NINFER_KV_REUSE_HOSTBACKED = '1'; $env:NINFER_TERNARY_PTQ1_FAST = '1'
  if ($SinkTokens -ne '') { $env:NINFER_KV_SINK = $SinkTokens }
  if ($Scorer -ne '')     { $env:NINFER_TERNARY_KVMEM = $Scorer }

  for ($i = 0; $i -lt 15; $i++) {
    $own = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
           Select-Object -First 1 -ExpandProperty OwningProcess
    if (-not $own) { break }
    Stop-Process -Id $own -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2
  }
  $dir = Join-Path $OutRoot ("{0}-{1}" -f $Section, $Name)
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $errLog = Join-Path $dir 'err.log'
  Remove-Item $errLog -ErrorAction SilentlyContinue
  $argv = @($Model,'--host','127.0.0.1','--port',"$port",'--model-id','qwen3.8-27b',
            '--kv-capacity',"$($PoolPages * 64)",'--max-context',"$MaxContext",
            '--max-concurrency','1','--no-thinking','--greedy','--host-kv-mib','16384')
  if ($RequestLog -ne '') { $argv += @('--request-log-jsonl', $RequestLog) }
  $argv += $Extra
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $p = Start-Process -FilePath $Exe -ArgumentList $argv -PassThru -NoNewWindow `
         -WorkingDirectory (Split-Path $Exe -Parent) `
         -RedirectStandardOutput (Join-Path $dir 'out.log') -RedirectStandardError $errLog
  $ready = $false; $dl = (Get-Date).AddSeconds($ReadySec)
  while ((Get-Date) -lt $dl) {
    if ($p.HasExited) { break }
    try { $r = Invoke-WebRequest ("http://127.0.0.1:{0}/health" -f $port) -UseBasicParsing -TimeoutSec 3
          if ($r.StatusCode -eq 200) { $ready = $true; break } } catch { Start-Sleep -Milliseconds 600 }
  }
  $http = ''
  if ($ready -and $SendRequest) {
    $body = @{ model = 'qwen3.8-27b'; messages = @(@{ role = 'user'; content = ('Summarise: ' + ('archive ledger line ' * 60)) }); max_tokens = 24; stream = $false }
    $bf = Join-Path $dir 'body.json'
    [System.IO.File]::WriteAllText($bf, ($body | ConvertTo-Json -Depth 6 -Compress), (New-Object System.Text.UTF8Encoding($false)))
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    $resp = & $curl -s -o (Join-Path $dir 'resp.json') -w '%{http_code}' -X POST -H 'Content-Type: application/json' `
              --data-binary ("@" + $bf) ("http://127.0.0.1:{0}/v1/chat/completions" -f $port) 2>&1
    $http = ($resp -join '').Trim()
    Start-Sleep -Seconds 2
  }
  if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Start-Sleep -Milliseconds 800 }
  $own = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
         Select-Object -First 1 -ExpandProperty OwningProcess
  if ($own) { Stop-Process -Id $own -Force -ErrorAction SilentlyContinue }
  $sw.Stop()

  $txt = ''
  if (Test-Path $errLog) { $txt = Get-Content -Raw $errLog }
  $poolLine = (Select-String -Path $errLog -Pattern 'capacity \| KV' -EA SilentlyContinue | Select-Object -First 1).Line
  $runtime = ''; $free = ''; $pages = ''
  $m = [regex]::Match([string]$poolLine, 'pages\s+([\d,]+)/[\d,]+\s+\|\s+runtime\s+([\d.]+)\s*([KMGT]iB)\s+\|\s+free\s+([\d.]+)\s*([KMGT]iB)')
  if ($m.Success) {
    $pages = $m.Groups[1].Value
    $runtime = [math]::Round((To-Mib $m.Groups[2].Value $m.Groups[3].Value), 1)
    $free = [math]::Round((To-Mib $m.Groups[4].Value $m.Groups[5].Value), 1)
  }
  $budgets = (Select-String -Path $errLog -Pattern '\[ring\] budgets:' -EA SilentlyContinue | Select-Object -First 1).Line
  $skel = ''
  if ([string]$budgets -match 'skeleton=(\d+)') { $skel = $matches[1] }
  $ringSeen = ([regex]::Matches($txt, '\[ring\] budgets:')).Count
  $guardHit = ([regex]::Matches($txt, 'leaves no working room in the KV pool')).Count
  $okLine = ([regex]::Matches($txt, 'engine ready')).Count

  # post-workload memory summary from the request log (T4/T5 of the batch)
  $mem = ''
  if ($RequestLog -ne '' -and (Test-Path $RequestLog)) {
    $lines = Get-Content $RequestLog -EA SilentlyContinue
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
      try { $o = $lines[$i] | ConvertFrom-Json } catch { continue }
      if ($o.memory) {
        $mm = $o.memory
        $mem = ("after_weights={0} after_startup={1} ws_cap={2} ws_peak={3} seq_cap={4} runtime_res={5} min_runtime={6} slack={7}" -f `
                $mm.available_after_weights_bytes, $mm.available_after_startup_bytes, $mm.workspace.capacity_bytes,
                $mm.workspace.peak_used_bytes, $mm.sequence.capacity_bytes, $mm.runtime_reservation_bytes,
                $mm.minimum_runtime_reservation_bytes, $mm.planned_slack_bytes)
        break
      }
    }
  }

  $row = [pscustomobject]@{
    section=$Section; name=$Name; pool_pages=$PoolPages; max_context=$MaxContext; sink=$SinkTokens
    scorer=$Scorer; extra=($Extra -join ' '); ready=[int]$ready; http=$http
    pages=$pages; runtime_mib=$runtime; free_mib=$free; skeleton_pages=$skel; ring_lines=$ringSeen
    engine_ready=$okLine; guard_refusal=$guardHit; secs=[math]::Round($sw.Elapsed.TotalSeconds,1); memory=$mem
  }
  $rows.Add($row) | Out-Null
  Save-Rows
  Say ("  {0,-10} {1,-22} ready={2,-5} http={3,-4} pages={4,-5} runtime={5,-8} free={6,-8} skel={7,-4} guard={8}" -f `
       $Section, $Name, $ready, $http, $pages, $runtime, $free, $skel, $guardHit)
  if ($mem -ne '') { Say ("      memory: " + $mem) }
  return $row
}

Say "=== A-tier batch START ==="
Say ("exe   = " + $Exe)
Say ("exe sha256 = " + (Get-FileHash $Exe -Algorithm SHA256).Hash)
Say ("model = " + $Model)
Say ("stray = " + @(Get-Process -Name ninfer,ninfer-serve -EA SilentlyContinue).Count)

# ---------------------------------------------------------------- T2: three-lever combination
Say ""
Say "### T2 three-lever combination (pool 64 pages) ###"
Start-Arm -Section 'T2' -Name 'base-chunk1024-graph-gdn32' -PoolPages 64 | Out-Null
Start-Arm -Section 'T2' -Name 'gdn-fp16'   -PoolPages 64 -Extra @('--gdn-state-fp16') | Out-Null
Start-Arm -Section 'T2' -Name 'chunk-256'  -PoolPages 64 -Extra @('--prefill-chunk','256') | Out-Null
Start-Arm -Section 'T2' -Name 'no-graph'   -PoolPages 64 -Extra @('--no-cuda-graph') | Out-Null
Start-Arm -Section 'T2' -Name 'all-three'  -PoolPages 64 -Extra @('--gdn-state-fp16','--prefill-chunk','256','--no-cuda-graph') | Out-Null

# ---------------------------------------------------------------- T3: the two sink copies agree
Say ""
Say "### T3 sink two-copy consistency (SINK=640 tokens = 10 pages) ###"
Start-Arm -Section 'T3' -Name 'sink640-pool64' -PoolPages 64 -SinkTokens '640' | Out-Null
Start-Arm -Section 'T3' -Name 'sink1920-pool64' -PoolPages 64 -SinkTokens '1920' | Out-Null

# ---------------------------------------------------------------- T4: lean + request log
Say ""
Say "### T4 lean profile + --request-log-jsonl (post-workload memory) ###"
$lean = @('--gdn-state-fp16','--prefill-chunk','128','--no-cuda-graph')
$rl1 = Join-Path $OutRoot 'reqlog-pool34.jsonl'
$rl2 = Join-Path $OutRoot 'reqlog-pool512.jsonl'
Remove-Item $rl1,$rl2 -ErrorAction SilentlyContinue
Start-Arm -Section 'T4' -Name 'lean-pool34'  -PoolPages 34  -Extra $lean -SendRequest -RequestLog $rl1 | Out-Null
Start-Arm -Section 'T4' -Name 'lean-pool512' -PoolPages 512 -Extra $lean -SendRequest -RequestLog $rl2 | Out-Null

# ---------------------------------------------------------------- T5/T6: reuse the existing sweep
Say ""
Say "### T5 kMin threshold repeat (reuse probe-kmin-sweep.ps1; sink=2816 = 44 pages) ###"
& $KminSweep -Exe $Exe -Model $Model -Pools @(58,59,60,61) -SinkTokens 2816 -Root (Join-Path $OutRoot 'kmin-repeat-58-61')
Say ""
Say "### T6 17920 delivered tier tool visibility (pool 280 pages, sink=2816) ###"
& $KminSweep -Exe $Exe -Model $Model -Pools @(280) -SinkTokens 2816 -Root (Join-Path $OutRoot 'tier-17920')

# ---------------------------------------------------------------- T7: max-context == pool
Say ""
Say "### T7 --max-context exactly equal to the pool (1024 pages = 65536 tokens) ###"
Start-Arm -Section 'T7' -Name 'maxctx-eq-pool' -PoolPages 1024 -MaxContext 65536 -Extra @('--gdn-state-fp16','--prefill-chunk','128','--no-cuda-graph') | Out-Null

# ---------------------------------------------------------------- T1: the full suite, LAST
if (-not $SkipFullTests) {
  Say ""
  Say "### T1 FULL ctest suite (258) -- long; artifacts, not Write-Host capture ###"
  $tRoot = Join-Path $OutRoot 'tests-full'
  New-Item -ItemType Directory -Force -Path $tRoot | Out-Null
  & $TestsScript -OutRoot $tRoot -Subset '.' | Out-Null
  $tr = Join-Path $tRoot 'test-run.tsv'
  if (Test-Path $tr) {
    $row = Import-Csv $tr -Delimiter "`t" | Select-Object -First 1
    Say ("  full suite: run={0} passed={1} failed={2} skipped={3} verdict={4}" -f $row.subset_run,$row.passed,$row.failed,$row.skipped,$row.subset_verdict)
    $xml = Join-Path $tRoot 'ctest-subset.xml'
    if (Test-Path $xml) {
      try {
        $doc = New-Object System.Xml.XmlDocument; $doc.Load($xml)
        $suites = @(); if ($doc.testsuites) { $suites = @($doc.testsuites.testsuite) } elseif ($doc.testsuite) { $suites = @($doc.testsuite) }
        $fails = @()
        foreach ($s in $suites) { foreach ($c in $s.testcase) { if ($c.failure) { $fails += ($c.name + ' [' + $c.failure.message + ']') } } }
        Say ("  failing tests = " + $fails.Count)
        $fails | Select-Object -First 20 | ForEach-Object { Say ("    " + $_) }
      } catch { Say ("  WARN: could not parse " + $xml + ": " + $_.Exception.Message) }
    }
  } else { Say "  WARN: test-run.tsv missing -- read the console log under " }
}

Say ""
Say "=== A-tier batch END ==="
Say ("rows = " + $rows.Count + "   tsv = " + $tsv)
Say ("stray = " + @(Get-Process -Name ninfer,ninfer-serve -EA SilentlyContinue).Count)
Say ("gpu used = " + ((& nvidia-smi --query-gpu=memory.used --format=csv,noheader) -join ''))
