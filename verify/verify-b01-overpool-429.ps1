# B01 saturation retest on the 8 GB tier.
#
# WHAT IT PROVES (ledger: handoff doc "KVMem line, new-window takeover" 20261007, line 62)
#   new engine  -> the request that cannot fit is REFUSED with HTTP 429 server_overloaded and a
#                  numbered message, and the engine KEEPS SERVING (a later request is still 200).
#   factory     -> worker crash + HTTP 500 + 503 cascade.
#
# HOW IT GOES RED (negative control)
#   Run the SAME script with -Exe <factory binary>. A verdict of PASS there means the instrument
#   cannot tell the two engines apart and this file is worthless. Run both arms, compare.
#   Second red control: -ShortProbeOnly (skip the long prompt) -> the refusal never happens, so
#   "429 observed" must come out False. If it still says True, the instrument is broken.
#
# SHAPE TAKEN FROM MEASUREMENT, NOT FROM A GUESS
#   8 GB tier real pool = --kv-capacity 2176 tokens = 34 pages (64 tokens/page).
#   Skeleton = 2808 tokens = 44 pages > 34 pages, so a long prompt CANNOT fit at all.
#   Measured in experiment E6: "a 42,357-token prompt against pool 2,176" is exactly the shape.
#   Guard negative control already measured: NINFER_KV_SINK=2112 (33 pages) lets it start.
#
# USAGE
#   pwsh -File b01-8gb-saturation.ps1 -Exe <path to ninfer-serve.exe> -Tag new
#   pwsh -File b01-8gb-saturation.ps1 -Exe <factory exe>              -Tag factory
#
# Pure ASCII on purpose: Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI, so non-ASCII
# comments silently corrupt the script.

param(
  [Parameter(Mandatory = $true)][string]$Exe,
  # Mandatory: pass the .ninfer artifact to measure. No default on purpose -- a published script
  # must not carry the author's model paths.
  [Parameter(Mandatory = $true)][string]$Model,
  [string]$Tag = 'new',
  [int]$Port = 8160,
  [int]$KvCapacity = 2176,        # 34 pages, the measured 8 GB tier pool
  [int]$MaxContext = 65536,       # logical 1024 pages
  [int]$SinkTokens = 2112,        # 33 pages: measured negative control for the startup guard
  [int]$HostKvMib = 16384,
  [int]$LongTargetChars = 150000, # ~42,357 tokens of English text per the measured shape
  [int]$ReadyTimeoutSec = 300,
  [switch]$ShortProbeOnly,
  [ValidateSet('ours', 'factory')][string]$ArgvProfile = 'ours',
  [switch]$ExpectStartRefusal,     # same-binary red control for the startup guard: sink > pool must be refused
  [int]$FactoryMaxContext = 204800,
  [int]$FactoryKvmemBudget = 36864,
  [int]$FactoryKvmemReserve = 16384,
  [int]$FactoryKvmemHostMib = 12288,
  # MEASURED 2026-10-07: the factory server refuses to start unless its own invariants hold --
  #   "KVMem requires aligned B/R, nonzero H, prefill <= R, and device capacity B+R <= max-context"
  # so shrinking its pool to match ours needs R >= prefill-chunk (and prefill must be a multiple of
  # 128). B=2112/R=64 with prefill 256 was rejected with that message plus a help dump, which cost a
  # round. B=2048/R=128/prefill=128 gives a 2176-token pool -- identical to our 8 GB tier arm.
  [int]$FactoryPrefillChunk = 256,
  [string]$OutRoot = (Join-Path $PSScriptRoot 'out-b01'),
  # CUDA runtime DLLs must be on PATH or the engine exits with 0xC0000135 (STATUS_DLL_NOT_FOUND)
  # before printing anything. Pass -CudaRoot to prepend that install's bin directories.
  [string]$CudaRoot = $env:CUDA_PATH
)

$ErrorActionPreference = 'Continue'

function Say([string]$text) { Write-Host $text }

Say ("=== B01 8GB saturation retest  arm=" + $Tag + " ===")
Say ("exe = " + $Exe)
if (-not (Test-Path $Exe)) { Say "FATAL: exe not found"; exit 2 }
if (-not (Test-Path $Model)) { Say "FATAL: model not found"; exit 2 }
Say ("exe sha256 = " + (Get-FileHash $Exe -Algorithm SHA256).Hash)
Say ("exe size   = " + (Get-Item $Exe).Length + " bytes")
Say ("model      = " + $Model + "  (" + [int]((Get-Item $Model).Length / 1MB) + " MB)")

# ---- preflight: environment assertions before touching the GPU -------------------------------
Say ""
Say "--- preflight ---"
$stray = @(Get-Process -Name 'ninfer','ninfer-serve','ninfer-kvmem-server' -ErrorAction SilentlyContinue)
Say ("stray ninfer processes = " + $stray.Count)
if ($stray.Count -gt 0) {
  $stray | ForEach-Object { Say ("  pid=" + $_.Id + " name=" + $_.ProcessName) }
  Say "FATAL: another engine instance is running; refusing to start (GPU is exclusive)"
  exit 3
}
$owner = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess
Say ("port " + $Port + " owner pid = " + $(if ($owner) { $owner } else { "free" }))
if ($owner) { Say "FATAL: port in use"; exit 3 }
$smi = & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1
Say ("gpu memory = " + ($smi -join ' | '))

$dir = Join-Path $OutRoot $Tag
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$outLog = Join-Path $dir 'out.log'
$errLog = Join-Path $dir 'err.log'
$startStamp = Get-Date
$startStamp.ToString('o') | Set-Content (Join-Path $dir 'start.txt')

# ---- environment: the measured 8 GB launcher configuration ----------------------------------
$env:NINFER_KV_SINK = "$SinkTokens"
$env:NINFER_KV_RING = '1'
$env:NINFER_KV_WINDOW = '1536'
$env:NINFER_KV_RETRIEVE = '768'
$env:NINFER_HOST_PAGEABLE = '1'
$env:NINFER_KV_REUSE_HOSTBACKED = '1'
$env:NINFER_TERNARY_PTQ1_FAST = '1'
$exeDir = Split-Path $Exe -Parent
$env:PATH = $exeDir + ';' + $env:PATH
if ($CudaRoot) {
  foreach ($sub in @('bin', 'bin\x64')) {
    $d = Join-Path $CudaRoot $sub
    if (Test-Path $d) { $env:PATH = $d + ';' + $env:PATH }
  }
}

# Two engines, two argv dialects. The PROBES and the CRITERIA stay identical (that is what makes the
# comparison meaningful); only the launch recipe differs, because the factory binary accepts the
# --kvmem-* family and ours does not (measured: "--kvmem-" has 0 hits tree-wide in our source).
$argv = if ($ArgvProfile -eq 'factory') {
  @(
    $Model,
    '--host', '127.0.0.1',
    '--port', "$Port",
    # MEASURED 2026-10-07: without --model-id the factory server advertises the model's own id
    # (e.g. 'qwen3.8-27b-gsq-rco-iq3s') and every request naming anything else gets HTTP 404
    # "model '...' not found" -- which my first factory arm mistook for engine behaviour.
    '--model-id', 'qwen3.8-27b',
    '--max-context', "$FactoryMaxContext",
    '--max-concurrency', '1',
    '--prefill-chunk', "$FactoryPrefillChunk",
    '--default-max-tokens', '16384',
    '--kvmem-budget', "$FactoryKvmemBudget",
    '--kvmem-gen-reserve', "$FactoryKvmemReserve",
    '--kvmem-host-mib', "$FactoryKvmemHostMib",
    '--kvmem-sessions', '1',
    '--kv-dtype', 'int8',
    '--device', '0',
    '--device-profile', 'off',
    '--spec', 'mtp',
    '--draft-tokens', '4',
    '--adaptive-mtp',
    '--ngram-draft-tokens', '31'
  )
} else {
  @(
    $Model,
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--model-id', 'qwen3.8-27b',
    '--kv-capacity', "$KvCapacity",
    '--max-context', "$MaxContext",
    '--max-concurrency', '1',
    '--no-thinking',
    '--greedy',
    '--host-kv-mib', "$HostKvMib",
    # NOTE: the older probe script this shape was taken from passed --no-kv-lease-growth. That flag
    # DOES NOT EXIST in this tree: serve_options.cpp has only --kv-lease-growth (:922-923, which sets
    # it to TRUE) and include/ninfer/types.h:321 declares the default as false, pinned by
    # tests/test_serve_options.cpp:1370-1371. Worse, an unrecognised --flag is not rejected here --
    # the parse chain ends at ":992 } else if (parse_dispatch_options(arg)) { }" with no else-throw --
    # so a wrong flag is SILENTLY IGNORED and a run can look green while configured differently.
    # Omitting it is exactly "no lease growth"; passing a made-up flag would prove nothing.
    '--prefill-chunk', '128',
    '--gdn-state-fp16',
    '--no-cuda-graph'
  )
}
Say ""
Say "--- launch ---"
Say ("argv: " + ($argv -join ' '))
Push-Location $exeDir
$proc = Start-Process -FilePath $Exe -ArgumentList $argv -PassThru -NoNewWindow `
  -RedirectStandardOutput $outLog -RedirectStandardError $errLog
Pop-Location
Say ("pid = " + $proc.Id)

# ---- startup-guard control: SAME binary, so it is a legitimate same-binary red control ---------
# Positive control shape measured earlier: NINFER_KV_SINK=2816 (44 pages) against pool 2176
# (34 pages) => exit code 1 plus a full message. If this comes out green while the real 8 GB sink
# is 2112 (33 pages, which must START), the guard is either not wired or my argv is not reaching it.
if ($ExpectStartRefusal) {
  $guardDeadline = (Get-Date).AddSeconds(240)
  while (-not $proc.HasExited -and (Get-Date) -lt $guardDeadline) { Start-Sleep -Seconds 2 }
  $exited = $proc.HasExited
  $code = $null
  if ($exited) { try { $code = $proc.ExitCode } catch { $code = 'unknown' } }
  $guardText = ''
  if (Test-Path $errLog) {
    $g = Select-String -Path $errLog -Pattern 'sink|skeleton|pool|Current configuration' -ErrorAction SilentlyContinue | Select-Object -First 8
    $guardText = (($g | ForEach-Object { $_.Line.Trim() }) -join ' | ')
  }
  Say ("GUARD CONTROL: exited=" + $exited + " exit_code=" + $(if ($null -eq $code) { "-" } else { $code }))
  Say ("GUARD CONTROL: message = " + $guardText)
  $guardOk = $exited -and ($code -ne 0) -and ($code -ne 'unknown') -and ($guardText.Length -gt 0)
  Say ("VERDICT guard-start-refusal = " + $(if ($guardOk) { 'PASS' } else { 'FAIL' }))
  if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
  exit $(if ($guardOk) { 0 } else { 5 })
}

# ---- wait for /health ------------------------------------------------------------------------
$deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
$ready = $false
while ((Get-Date) -lt $deadline) {
  if ($proc.HasExited) { break }
  try {
    $r = Invoke-WebRequest ("http://127.0.0.1:" + $Port + "/health") -UseBasicParsing -TimeoutSec 5
    if ($r.StatusCode -eq 200) { $ready = $true; break }
  } catch { Start-Sleep -Seconds 3 }
}
Say ("ready = " + $ready + "   exited = " + $proc.HasExited)
if (-not $ready) {
  Say "--- err.log tail ---"
  if (Test-Path $errLog) { Get-Content $errLog -Tail 25 | ForEach-Object { Say ("  " + $_) } }
  Say "VERDICT arm=$Tag  FAIL(startup)"
  if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
  exit 4
}

function Post-Chat([string]$content, [int]$maxTokens = 64) {
  # curl.exe, NOT Invoke-WebRequest. MEASURED 2026-10-07: on this box PowerShell 5.1 threw for the 429
  # and $_.Exception.Response.GetResponseStream() yielded an EMPTY body, so the client-visible message
  # came out blank and the "message carries a number" criterion read False on a run that was actually
  # correct. curl prints the body whatever the status code is, so the criterion is decided by the
  # engine instead of by the HTTP client's error handling.
  $payload = @{
    model = 'qwen3.8-27b'
    messages = @(@{ role = 'user'; content = $content })
    max_tokens = $maxTokens
    temperature = 0
  } | ConvertTo-Json -Depth 8 -Compress
  $stamp = [guid]::NewGuid().ToString('N')
  $payloadFile = Join-Path $dir ('payload-' + $stamp + '.json')
  $bodyFile = Join-Path $dir ('body-' + $stamp + '.txt')
  [System.IO.File]::WriteAllText($payloadFile, $payload, (New-Object System.Text.UTF8Encoding($false)))
  $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $status = 0
  $err = ''
  if (Test-Path $curl) {
    $codeText = & $curl -s -o $bodyFile -w '%{http_code}' -H 'Content-Type: application/json' --data-binary ('@' + $payloadFile) --max-time 900 ('http://127.0.0.1:' + $Port + '/v1/chat/completions') 2>&1
    $status = 0
    [void][int]::TryParse(([string]$codeText).Trim(), [ref]$status)
  } else {
    $err = 'curl.exe not found'
  }
  $sw.Stop()
  $body = ''
  if (Test-Path $bodyFile) { $body = [System.IO.File]::ReadAllText($bodyFile) }
  $msg = ''
  if ($body) {
    try { $msg = [string](($body | ConvertFrom-Json).error.message) } catch { $msg = $body }
  }
  [pscustomobject]@{
    status = $status; elapsed_s = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    message = ($msg -replace '\s+', ' ').Trim(); transport_error = $err
    body_file = $bodyFile
  }
}

# ---- long prompt: the request that cannot fit -------------------------------------------------
Say ""
Say "--- phase A: long prompt (~" + $LongTargetChars + " chars, target >= 42000 tokens) ---"
$long = $null
if ($ShortProbeOnly) {
  Say "  SKIPPED (-ShortProbeOnly): this is the red control, 'refused' must come out False"
} else {
  $sentence = 'The quick brown archive line records mundane maintenance notes for the ledger. '
  $reps = [int][math]::Ceiling($LongTargetChars / $sentence.Length)
  $big = ($sentence * $reps) + "`n`nReply with the single word: done."
  $long = Post-Chat $big 64
  Say ("  HTTP " + $long.status + "  in " + $long.elapsed_s + "s")
  Say ("  message: " + $long.message)
  if ($long.transport_error) { Say ("  transport: " + $long.transport_error) }
}

# ---- short prompt: THE criterion that the engine is still serving -----------------------------
Say ""
Say "--- phase B: short prompt (engine must still serve) ---"
$short = Post-Chat 'Reply with 42 only.' 16
Say ("  HTTP " + $short.status + "  in " + $short.elapsed_s + "s")
Say ("  message: " + $short.message)

# ---- observability surfaces ------------------------------------------------------------------
Say ""
Say "--- phase C: /metrics + /v1/models ---"
$metricLine = ''
try {
  $m = Invoke-WebRequest ("http://127.0.0.1:" + $Port + "/metrics") -UseBasicParsing -TimeoutSec 10
  $hit = ($m.Content -split "`n") | Where-Object { $_ -match 'context_cache_exhausted' } | Select-Object -First 3
  if ($hit) { $hit | ForEach-Object { Say ("  metric: " + $_.Trim()) } } else { Say "  no context_cache_exhausted line in /metrics" }
  $metricLine = ($hit -join ' ; ')
} catch { Say ("  /metrics not available: " + $_.Exception.Message) }
try {
  $models = Invoke-WebRequest ("http://127.0.0.1:" + $Port + "/v1/models") -UseBasicParsing -TimeoutSec 10
  Say ("  /v1/models HTTP " + [int]$models.StatusCode + "  (NOTE: 200 here does NOT prove the engine can serve)")
} catch { Say ("  /v1/models failed: " + $_.Exception.Message) }

# ---- server-side readings --------------------------------------------------------------------
Say ""
Say "--- server stderr readings ---"
$crashed = $proc.HasExited
$exitCode = $null
if ($crashed) { try { $exitCode = $proc.ExitCode } catch { $exitCode = 'unknown' } }
Say ("  server exited = " + $crashed + "  exit code = " + $(if ($null -eq $exitCode) { "-" } else { $exitCode }))
$budgets = ''
if (Test-Path $errLog) {
  # SELF-REPORT, not a copy of my own argv: serve_options.cpp does not reject an unrecognised --flag
  # (the chain ends at ":992 } else if (parse_dispatch_options(arg)) { }" with no else-throw), so a
  # flag that does not exist is silently ignored and this startup line is the only proof of what the
  # server actually ran with. Compare it against the argv printed above; a mismatch means a flag was
  # swallowed and the run proves nothing.
  $cfg = Select-String -Path $errLog -Pattern 'INFO  capacity \| KV|Current configuration|KV capacity' -ErrorAction SilentlyContinue | Select-Object -First 4
  if ($cfg) { Say "  [effective config, self-reported]" ; $cfg | ForEach-Object { Say ("    " + $_.Line.Trim()) } }
  else { Say "  [effective config] no 'capacity | KV' self-report line in stderr -- cannot prove the flags took effect" }
  $bl = Select-String -Path $errLog -Pattern '\[ring\] budgets:' -ErrorAction SilentlyContinue | Select-Object -Last 1
  if ($bl) { $budgets = $bl.Line.Trim(); Say ("  " + $budgets) }
  $bad = Select-String -Path $errLog -Pattern 'worker crash|out of memory|ContextCacheExhausted|context cache exhausted|unhandled|terminate|abort' -ErrorAction SilentlyContinue
  Say ("  crash/OOM/exhaustion lines = " + @($bad).Count)
  $bad | Select-Object -First 12 | ForEach-Object { Say ("    " + $_.Line.Trim()) }
}

# ---- verdict ---------------------------------------------------------------------------------
$refused = ($null -ne $long) -and ($long.status -eq 429)
$numbered = ($null -ne $long) -and ($long.message -match '\d')
$stillServing = ($short.status -eq 200)
$verdict = if ($refused -and $numbered -and $stillServing -and (-not $crashed)) { 'PASS' }
           elseif ($ShortProbeOnly) { 'PASS(red-control: no refusal expected)' }
           else { 'FAIL' }

Say ""
Say "=== VERDICT arm=$Tag  $verdict ==="
Say ("  refused with 429          = " + $refused)
Say ("  message carries a number  = " + $numbered)
Say ("  engine still serving (200)= " + $stillServing)
Say ("  server survived           = " + (-not $crashed))

$row = [pscustomobject]@{
  tag = $Tag; exe = $Exe; exe_sha256 = (Get-FileHash $Exe -Algorithm SHA256).Hash
  port = $Port; kv_capacity = $KvCapacity; sink = $SinkTokens; max_context = $MaxContext
  long_status = $(if ($null -ne $long) { $long.status } else { 'skipped' })
  long_message = $(if ($null -ne $long) { $long.message } else { '' })
  short_status = $short.status; short_message = $short.message
  server_exited = $crashed; exit_code = $exitCode; budgets = $budgets
  metrics = $metricLine; verdict = $verdict
  started = $startStamp.ToString('o'); finished = (Get-Date).ToString('o')
}
$tsv = Join-Path $OutRoot ($Tag + '-result.tsv')
$row | Export-Csv -Path $tsv -Delimiter "`t" -NoTypeInformation -Encoding UTF8
Say ("WROTE " + $tsv)

if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force; Start-Sleep -Seconds 2 }
$left = @(Get-Process -Name 'ninfer','ninfer-serve','ninfer-kvmem-server' -ErrorAction SilentlyContinue)
Say ("cleanup: remaining ninfer processes = " + $left.Count)
