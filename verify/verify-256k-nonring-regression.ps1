# 256k non-ring HTTP regression: starts our engine, then runs PR #2's own verification script.
#
# WHAT IT VERIFIES
#   PR #2 ("respect physical KV reservations when ring mode is disabled") changes code paths that only
#   execute when the KVMem ring is OFF. Its own script (verify/kv-reservation-off-regression.py) is the
#   instrument: it sends a ~250k-token document and asserts the three planted access codes come back,
#   then runs 4 concurrent lanes for two turns and asserts a cached follow-up (cached_tokens > 0).
#
# CRITERIA
#   script exit code 0, and the JSON report's every entry has passed=true.
#   The script itself fails the near_256k phase unless prompt_tokens lands in [250000, 262144].
#
# HOW IT GOES RED (two controls, both on the same binary)
#   -RedControl   : same server, same script, but --long-rows 200 (a tiny document). The
#                   prompt_tokens criterion MUST then fail. If it still exits 0, this instrument is
#                   measuring nothing.
#   kvmem-off assertion: the script is only meaningful with the ring disabled, so this launcher
#                   REMOVES every NINFER_KV_* / NINFER_HOST_PAGEABLE variable and asserts they are gone.
#                   If the assertion cannot see them at all, the check is vacuous and says so.
#
# USAGE
#   pwsh -File run-256k-pr2.ps1 -Exe <build>\apps\ninfer-serve.exe
#   pwsh -File run-256k-pr2.ps1 -Exe ... -RedControl
#
# Pure ASCII on purpose (Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI).

param(
  [Parameter(Mandatory = $true)][string]$Exe,
  # Mandatory on purpose: the intended artifact lives under a path with non-ASCII characters, and a
  # hard-coded default here would make this file non-ASCII -- which Windows PowerShell 5.1 then reads
  # as ANSI and mangles. Pass the real path in.
  # Intended model: the 9,079 MB Ternary-Bonsai-2-27B-ninfer-v3.ninfer in the model-package directory.
  [Parameter(Mandatory = $true)][string]$Model,
  # PR #2's own instrument ships beside this launcher in verify/.
  [string]$Script = (Join-Path $PSScriptRoot 'kv-reservation-off-regression.py'),
  # 'python' resolves from PATH; pass an absolute interpreter path if that is not reliable.
  [string]$PythonExe = 'python',
  [string]$ModelId = 'qwen3.8-27b',
  [int]$Port = 8180,
  [int]$MaxContext = 262144,
  [int]$KvCapacity = 262144,
  [int]$MaxConcurrency = 4,
  [string]$KvDtype = 'int8',
  [int]$PrefillChunk = 1024,
  [int]$LongRows = 15700,
  [int]$ConcurrentRows = 6800,
  [int]$ReadyTimeoutSec = 600,
  [switch]$RedControl,
  # Optional label so a caller can name the arm; without it the arm is named from -RedControl.
  # (A caller that passes -Tag to a script lacking it dies instantly with "A parameter cannot be
  # found" and 0 minutes elapsed -- which is exactly what happened on the first attempt to run this.)
  [string]$Tag = '',
  [string]$OutRoot = (Join-Path $PSScriptRoot 'out-256k'),
  [string]$CudaRoot = $env:CUDA_PATH
)

$ErrorActionPreference = 'Continue'

function Say([string]$t) { Write-Host $t }

$tag = if ($Tag -ne '') { $Tag } elseif ($RedControl) { 'redcontrol' } else { 'main' }
Say ("=== 256k non-ring regression  arm=" + $tag + " ===")
if (-not (Test-Path $Exe)) { Say "FATAL: exe not found: $Exe"; exit 2 }
if (-not (Test-Path $Model)) { Say "FATAL: model not found: $Model"; exit 2 }
if (-not (Test-Path $Script)) { Say "FATAL: PR script not found: $Script"; exit 2 }
if (-not (Test-Path $PythonExe)) { Say "FATAL: python not found: $PythonExe"; exit 2 }
Say ("exe sha256 = " + (Get-FileHash $Exe -Algorithm SHA256).Hash)
Say ("model      = " + $Model + "  (" + [int]((Get-Item $Model).Length / 1MB) + " MB)")
Say ("script     = " + $Script)
Say ("script sha = " + (Get-FileHash $Script -Algorithm SHA256).Hash.Substring(0, 16))

# ---- preflight ---------------------------------------------------------------------------------
Say ""
Say "--- preflight ---"
$stray = @(Get-Process -Name 'ninfer', 'ninfer-serve', 'ninfer-kvmem-server' -ErrorAction SilentlyContinue)
Say ("stray ninfer processes = " + $stray.Count)
if ($stray.Count -gt 0) {
  $stray | ForEach-Object { Say ("  pid=" + $_.Id + " " + $_.ProcessName) }
  Say "FATAL: another engine instance is running; the GPU is exclusive"
  exit 3
}
$owner = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess
if ($owner) { Say ("FATAL: port $Port already owned by pid $owner"); exit 3 }
Say ("port $Port free")
$smi = & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1
Say ("gpu memory = " + ($smi -join ' | '))

$dir = Join-Path $OutRoot $tag
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$outLog = Join-Path $dir 'server-out.log'
$errLog = Join-Path $dir 'server-err.log'
$report = Join-Path $dir 'pr2-report.json'
$startStamp = Get-Date
$startStamp.ToString('o') | Set-Content (Join-Path $dir 'start.txt')

# ---- KVMem OFF: remove, then ASSERT they are gone (a vacuous check must be visible) ------------
$kvVars = @('NINFER_KV_SINK', 'NINFER_KV_RING', 'NINFER_KV_WINDOW', 'NINFER_KV_RETRIEVE',
            'NINFER_KV_REUSE_HOSTBACKED', 'NINFER_KV_RETRIEVE_SHARE', 'NINFER_HOST_PAGEABLE',
            'NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL', 'NINFER_TERNARY_KVMEM_SCORE_MAXQ')
$seenBefore = @($kvVars | Where-Object { Test-Path ("env:" + $_) })
Say ""
Say ("--- ring OFF assertion ---")
Say ("  KVMem-ish vars visible before removal = " + $seenBefore.Count + " " + ($seenBefore -join ','))
foreach ($v in $kvVars) { Remove-Item ("env:" + $v) -ErrorAction SilentlyContinue }
$stillSet = @($kvVars | Where-Object { Test-Path ("env:" + $_) })
Say ("  still set after removal = " + $stillSet.Count + " (must be 0)")
if ($stillSet.Count -ne 0) { Say "FATAL: cannot disable the ring; the run would test the wrong path"; exit 4 }

$argv = @(
  $Model,
  '--host', '127.0.0.1',
  '--port', "$Port",
  '--model-id', $ModelId,
  '--max-context', "$MaxContext",
  '--kv-capacity', "$KvCapacity",
  '--kv-dtype', $KvDtype,
  '--max-concurrency', "$MaxConcurrency",
  '--prefill-chunk', "$PrefillChunk",
  '--no-thinking',
  '--greedy'
)
Say ""
Say "--- launch ---"
Say ("argv: " + ($argv -join ' '))
$exeDir = Split-Path $Exe -Parent
$env:PATH = $exeDir + ';' + $env:PATH
if ($CudaRoot) {
  foreach ($sub in @('bin', 'bin\x64')) {
    $d = Join-Path $CudaRoot $sub
    if (Test-Path $d) { $env:PATH = $d + ';' + $env:PATH }
  }
}
Push-Location $exeDir
$proc = Start-Process -FilePath $Exe -ArgumentList $argv -PassThru -NoNewWindow -RedirectStandardOutput $outLog -RedirectStandardError $errLog
Pop-Location
Say ("pid = " + $proc.Id)

$deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
$ready = $false
$loadSw = [System.Diagnostics.Stopwatch]::StartNew()
while ((Get-Date) -lt $deadline) {
  if ($proc.HasExited) { break }
  try { $r = Invoke-WebRequest ("http://127.0.0.1:$Port/health") -UseBasicParsing -TimeoutSec 5; if ($r.StatusCode -eq 200) { $ready = $true; break } } catch { Start-Sleep -Seconds 5 }
}
$loadSw.Stop()
Say ("ready = $ready   load seconds = " + [math]::Round($loadSw.Elapsed.TotalSeconds, 1) + "   exited = " + $proc.HasExited)
if (-not $ready) {
  Say "--- err.log tail ---"
  if (Test-Path $errLog) { Get-Content $errLog -Tail 30 | ForEach-Object { Say ("  " + $_) } }
  Say "VERDICT arm=$tag FAIL(startup)"
  if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
  exit 5
}
Say ("gpu memory after load = " + ((& nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>&1) -join ' | '))

# ---- run PR #2's script ------------------------------------------------------------------------
Say ""
Say "--- PR #2 regression script ---"
$scriptArgs = @($Script, '--url', ("http://127.0.0.1:" + $Port), '--model', $ModelId, '--report', $report,
                '--long-rows', "$LongRows", '--concurrent-rows', "$ConcurrentRows")
if ($RedControl) {
  $scriptArgs = @($Script, '--url', ("http://127.0.0.1:" + $Port), '--model', $ModelId, '--report', $report,
                  '--phase', 'long', '--long-rows', '200')
  Say "  RED CONTROL MODE: --phase long --long-rows 200 (the prompt_tokens criterion must fail)"
}
Say ("  python " + ($scriptArgs -join ' '))
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$pyOut = & $PythonExe @scriptArgs 2>&1
$pyRc = $LASTEXITCODE
$sw.Stop()
$pyOut | Set-Content -Path (Join-Path $dir 'pr2-script-console.txt') -Encoding utf8
$pyOut | ForEach-Object { Say ("  | " + $_) }
Say ("  script exit = $pyRc   wall = " + [math]::Round($sw.Elapsed.TotalMinutes, 1) + " min")

# ---- read the report ---------------------------------------------------------------------------
$allPassed = $null
if (Test-Path $report) {
  try {
    $j = Get-Content $report -Raw | ConvertFrom-Json
    $entries = @()
    foreach ($p in $j.PSObject.Properties) {
      $v = $p.Value
      if ($v -is [System.Array]) { foreach ($item in $v) { $entries += [pscustomobject]@{ phase = $p.Name; passed = $item.passed; status = $item.status; hits = $item.hits; ttft = $item.ttft_s; prompt_tokens = $item.usage.prompt_tokens; cached = $item.usage.prompt_tokens_details.cached_tokens } } }
      else { $entries += [pscustomobject]@{ phase = $p.Name; passed = $v.passed; status = $v.status; hits = $v.hits; ttft = $v.ttft_s; prompt_tokens = $v.usage.prompt_tokens; cached = $v.usage.prompt_tokens_details.cached_tokens } }
    }
    Say ""
    Say "--- report entries ---"
    $entries | ForEach-Object { Say ("  {0,-22} passed={1,-6} status={2,-5} hits={3,-4} prompt_tokens={4,-8} cached={5,-7} ttft={6}" -f $_.phase, $_.passed, $_.status, $_.hits, $_.prompt_tokens, $_.cached, $_.ttft) }
    $allPassed = (@($entries | Where-Object { $_.passed -ne $true }).Count -eq 0) -and ($entries.Count -gt 0)
    Say ("  entries = " + $entries.Count + "  all passed = " + $allPassed)
  } catch { Say ("  could not parse report: " + $_.Exception.Message) }
} else { Say ("  no report written at " + $report) }

# ---- server-side readings ----------------------------------------------------------------------
Say ""
Say "--- server readings ---"
$crashed = $proc.HasExited
Say ("  server exited = $crashed")
if (Test-Path $errLog) {
  $ev = Select-String -Path $errLog -Pattern 'KV capacity|pool |free after|\[ring\]|reservation|reserved' -ErrorAction SilentlyContinue | Select-Object -First 12
  $ev | ForEach-Object { Say ("  " + $_.Line.Trim()) }
  $bad = @(Select-String -Path $errLog -Pattern 'worker crash|out of memory|invalid argument|error' -ErrorAction SilentlyContinue)
  Say ("  error-ish lines = " + $bad.Count)
  $bad | Select-Object -First 8 | ForEach-Object { Say ("    " + $_.Line.Trim()) }
}
Say ("  gpu memory at end = " + ((& nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>&1) -join ' | '))

# ---- verdict -----------------------------------------------------------------------------------
$verdict = if ($RedControl) {
  if ($pyRc -ne 0) { 'PASS(red control went red as required)' } else { 'FAIL(red control stayed green: instrument is blind)' }
} else {
  if ($pyRc -eq 0 -and $allPassed -eq $true -and -not $crashed) { 'PASS' } else { 'FAIL' }
}
Say ""
Say ("=== VERDICT arm=$tag  $verdict ===")

$record = [pscustomobject]@{
  when = (Get-Date).ToString('o'); arm = $tag; exe = $Exe
  exe_sha256 = (Get-FileHash $Exe -Algorithm SHA256).Hash; model = $Model
  max_context = $MaxContext; kv_capacity = $KvCapacity; kv_dtype = $KvDtype
  max_concurrency = $MaxConcurrency; prefill_chunk = $PrefillChunk
  long_rows = $LongRows; concurrent_rows = $ConcurrentRows
  script_exit = $pyRc; all_passed = $allPassed; server_exited = $crashed
  load_seconds = [math]::Round($loadSw.Elapsed.TotalSeconds, 1)
  script_minutes = [math]::Round($sw.Elapsed.TotalMinutes, 1)
  verdict = $verdict; report = $report
}
$rowFile = Join-Path $OutRoot ($tag + '-result.tsv')
$record | Export-Csv -Path $rowFile -Delimiter "`t" -NoTypeInformation -Encoding UTF8
Say ("WROTE " + $rowFile)

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force; Start-Sleep -Seconds 3 }
$left = @(Get-Process -Name 'ninfer', 'ninfer-serve', 'ninfer-kvmem-server' -ErrorAction SilentlyContinue)
Say ("cleanup: remaining ninfer processes = " + $left.Count)
