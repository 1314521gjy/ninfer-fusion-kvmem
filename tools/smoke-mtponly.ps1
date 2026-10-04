# smoke-mtponly.ps1 -- does the STRIPPED PQ2 artifact actually load and answer?
#
# Why this exists: check_strip.py / check_refs.py read the container, and reading the engine source
# says `--spec mtp` without --lm-head-draft should bind text/output_head with both final_hidden
# inputs (load/text.cpp:167-171). None of that is proof the engine comes up -- only a run is.
#
# What it proves, in order:
#   1. the artifact loads (startup log: `loading weights`, `capacity`, `reuse host-backed`)
#   2. one question gets a sane, checkable answer (not the 1-char degeneration this project has
#      seen before -- that is why the answer is graded, not just counted)
#   3. a second turn reuses the prefix instead of re-prefilling it (the whole point of the ball)
#
# ASCII-only on purpose: powershell.exe 5.1 reads a BOM-less .ps1 as ANSI. Exit 0 = PASS.
param(
  [string]$Engine = 'E:\ninfer\engines\ninfer-build89-cfe56ffb\ninfer-serve.exe',
  [string]$Model  = 'E:\infer-build\_convert\Ternary-Bonsai-2-27B-ninfer-v3-mtponly.ninfer',
  [int]$Port      = 8097,
  [string]$LogDir = 'E:\infer-build\exp'
)

$ErrorActionPreference = 'Continue'
$fail = 0
function Check([string]$Name, [bool]$Ok, [string]$Detail) {
  $v = 'OK  '; if (-not $Ok) { $v = 'FAIL'; $script:fail++ }
  Write-Host ("[{0}] {1,-46} {2}" -f $v, $Name, $Detail)
}

# ---- 0. paths must exist before anything is started (a typo here reads as a silent pass) ----
if (-not (Test-Path -LiteralPath $Engine)) { Write-Host "REFUSE: engine not found: $Engine"; exit 3 }
if (-not (Test-Path -LiteralPath $Model))  { Write-Host "REFUSE: model not found: $Model";  exit 3 }
if (-not (Test-Path -LiteralPath $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outLog = Join-Path $LogDir "smoke-mtponly-$stamp.out.log"
$errLog = Join-Path $LogDir "smoke-mtponly-$stamp.err.log"
$modelBytes = (Get-Item -LiteralPath $Model).Length
Write-Host ("ENGINE = {0}" -f $Engine)
Write-Host ("MODEL  = {0}  ({1:N0} B = {2:N3} GiB)" -f $Model, $modelBytes, ($modelBytes / 1GB))
Write-Host ("PORT   = {0}   LOG = {1}" -f $Port, $errLog)
Write-Host ''

# ---- 1. the five ring switches: env-only, and RETRIEVE is the one that keeps answers honest --
$env:NINFER_KV_WINDOW          = '16384'
$env:NINFER_KV_RETRIEVE        = '8192'
$env:NINFER_KV_RING            = '1'
$env:NINFER_HOST_PAGEABLE      = '1'
$env:NINFER_KV_REUSE_HOSTBACKED = '1'

function Read-Log([string]$Path) {
  # The engine holds its redirect handles open; a plain Get-Content throws and yields "" silently,
  # which reads as "no errors" -- open with FileShare.ReadWrite instead.
  if (-not (Test-Path -LiteralPath $Path)) { return '' }
  try {
    $fs = New-Object IO.FileStream($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    $sr = New-Object IO.StreamReader($fs)
    $t = $sr.ReadToEnd(); $sr.Close(); $fs.Close(); return $t
  } catch { return '' }
}

# ---- 2. start the engine (--spec mtp, NO --lm-head-draft, NO --vision: see load.cpp:99-102) ----
$argv = @(
  "`"$Model`"",
  '--host', '127.0.0.1', '--port', "$Port", '--model-id', 'qwen3.8-27b',
  '--max-context', '262144', '--kv-capacity', '17920', '--kv-dtype', 'k8v4', '--host-kv-mib', '16384',
  '--prefill-chunk', '1024', '--spec', 'mtp', '--draft-tokens', '4',
  '--default-max-tokens', '32768', '--default-reasoning-effort', 'none', '--max-concurrency', '1',
  '--max-shared-prefixes', '0',
  '--presence-penalty', '0', '--temperature', '0.7', '--top-p', '0.9', '--top-k', '20'
) -join ' '
Write-Host ("ARGV   = {0}" -f $argv)
Write-Host ''

$proc = Start-Process -FilePath $Engine -ArgumentList $argv -PassThru -NoNewWindow `
          -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$t0 = Get-Date
$ready = $false
for ($i = 0; $i -lt 240; $i++) {
  Start-Sleep -Milliseconds 500
  if ($proc.HasExited) { break }
  try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v1/models" -UseBasicParsing -TimeoutSec 2
    if ($r.StatusCode -eq 200) { $ready = $true; break }
  } catch { }
}
$loadSec = ((Get-Date) - $t0).TotalSeconds
Check 'engine came up (/v1/models 200)' $ready ("{0:N1} s" -f $loadSec)

$logText = Read-Log $errLog
if (-not $ready) {
  Write-Host ''
  Write-Host '--- startup log tail (last 25 lines) ---'
  ($logText -split "`r?`n" | Where-Object { $_.Trim() -ne '' } | Select-Object -Last 25) | ForEach-Object { Write-Host ("  " + $_) }
  if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
  Write-Host ''
  Write-Host 'SMOKE_VERDICT=FAIL (engine did not start)'
  exit 1
}

# ---- 3. read back the lines that matter -------------------------------------------------------
$lines = $logText -split "`r?`n"
$weightsLine  = ($lines | Where-Object { $_ -match 'loading weights' }  | Select-Object -Last 1)
$capacityLine = ($lines | Where-Object { $_ -match 'capacity\s*\|' }   | Select-Object -Last 1)
$reuseLine    = ($lines | Where-Object { $_ -match 'host-backed' }     | Select-Object -Last 1)
foreach ($pair in @(@('loading weights', $weightsLine), @('capacity', $capacityLine), @('host-backed reuse', $reuseLine))) {
  Check ("log line: {0}" -f $pair[0]) ($null -ne $pair[1] -and $pair[1].Trim() -ne '') ("{0}" -f $(if ($pair[1]) { $pair[1].Trim() } else { '(absent)' }))
}

# ---- 4. turn 1: a prompt with a needle in the middle, answer is GRADED ------------------------
# The needle is DERIVED from the same expression that builds the list -- hardcoding it is how a
# grader silently grades the wrong thing (this project has already shipped one wrong-needle test).
$owners = @('Zhang Wei','Li Na','Wang Fang','Liu Yang','Chen Jie','Zhao Lei','Sun Min','Zhou Qiang')
$facts = @()
for ($n = 1; $n -le 40; $n++) {
  $facts += ("Record {0}: project code ALPHA-{1}, owner {2}, status finished." -f $n, ($n * 7), $owners[$n % 8])
}
$expect27 = $owners[27 % 8]
$expect13 = $owners[13 % 8]
Write-Host ("NEEDLES: record 27 -> '{0}'   record 13 -> '{1}'" -f $expect27, $expect13)
$body = ($facts -join "`n")
$q1 = "Here is a list of records.`n`n$body`n`nQuestion: who is the owner in record 27? Answer with the name only."
$q2 = 'And who is the owner in record 13? Answer with the name only.'

function Ask([string]$Prompt, [int]$MaxTok) {
  $payload = @{
    model = 'qwen3.8-27b'
    messages = @(@{ role = 'user'; content = $Prompt })
    max_tokens = $MaxTok
    stream = $false
    temperature = 0
  } | ConvertTo-Json -Depth 6 -Compress
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method POST `
            -ContentType 'application/json' -Body $payload -UseBasicParsing -TimeoutSec 300
  $sw.Stop()
  return @{ Sec = $sw.Elapsed.TotalSeconds; Json = ($resp.Content | ConvertFrom-Json) }
}

try {
  $a1 = Ask $q1 96
  $txt1 = [string]$a1.Json.choices[0].message.content
  $u1 = $a1.Json.usage
  Write-Host ''
  Write-Host ("TURN 1  {0:N2} s   prompt_tokens={1}  completion_tokens={2}" -f $a1.Sec, $u1.prompt_tokens, $u1.completion_tokens)
  if ($u1.completion_tokens -gt 0 -and $a1.Sec -gt 0) {
    Write-Host ("        decode ~{0:N1} tok/s (upper bound: TTFT not split out)" -f ($u1.completion_tokens / $a1.Sec))
  }
  Write-Host ("        answer: {0}" -f ($txt1 -replace "`r?`n", ' / '))
  Check 'turn 1 answer is not degenerate (>=2 chars)' ($txt1.Trim().Length -ge 2) ("{0} chars" -f $txt1.Trim().Length)
  $hit1 = $txt1 -match [regex]::Escape($expect27)
  Check ("turn 1 needle hit (answer says {0})" -f $expect27) $hit1 ("{0}" -f $(if ($hit1) { 'needle found' } else { 'NEEDLE MISSED' }))

  # ---- 5. turn 2: same prefix plus one more question -- must reuse, not re-prefill -------------
  $payload2 = @{
    model = 'qwen3.8-27b'
    messages = @(
      @{ role = 'user';      content = $q1 },
      @{ role = 'assistant'; content = $txt1 },
      @{ role = 'user';      content = $q2 }
    )
    max_tokens = 32
    stream = $false
    temperature = 0
  } | ConvertTo-Json -Depth 6 -Compress
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r2 = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method POST `
          -ContentType 'application/json' -Body $payload2 -UseBasicParsing -TimeoutSec 300
  $sw.Stop()
  $j2 = $r2.Content | ConvertFrom-Json
  $txt2 = [string]$j2.choices[0].message.content
  Write-Host ''
  Write-Host ("TURN 2  {0:N2} s   prompt_tokens={1}  completion_tokens={2}" -f $sw.Elapsed.TotalSeconds, $j2.usage.prompt_tokens, $j2.usage.completion_tokens)
  Write-Host ("        answer: {0}" -f ($txt2 -replace "`r?`n", ' / '))
  Check 'turn 2 answer is not degenerate (>=2 chars)' ($txt2.Trim().Length -ge 2) ("{0} chars" -f $txt2.Trim().Length)
  $hit2 = $txt2 -match [regex]::Escape($expect13)
  Check ("turn 2 needle hit (answer says {0})" -f $expect13) $hit2 ("{0}" -f $(if ($hit2) { 'needle found' } else { 'NEEDLE MISSED' }))

  # the cache reading is the point: a full re-prefill would show a ~20 s turn-2 TTFT
  $log2 = Read-Log $errLog
  $cacheLines = ($log2 -split "`r?`n") | Where-Object { $_ -match 'cache\s' } | Select-Object -Last 3
  Write-Host ''
  Write-Host '--- cache / reuse readings (last 3) ---'
  if ($cacheLines) { $cacheLines | ForEach-Object { Write-Host ("  " + $_.Trim()) } } else { Write-Host '  (none found)' }
  $reused = $false
  foreach ($l in $cacheLines) {
    if ($l -match '\((\d+(?:\.\d+)?)%\)') { if ([double]$Matches[1] -ge 90) { $reused = $true } }
  }
  Check 'turn 2 reused prefix (cache >= 90%)' $reused 'see readings above'
} catch {
  Check 'request round-trip' $false $_.Exception.Message
}

# ---- 6. teardown ------------------------------------------------------------------------------
Write-Host ''
if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force; Start-Sleep -Seconds 2 }
$alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
Check 'engine torn down' ($null -eq $alive) ("pid {0}" -f $proc.Id)
Write-Host ''
Write-Host ("LOG FILES: {0} | {1}" -f $outLog, $errLog)
if ($fail -eq 0) { Write-Host 'SMOKE_VERDICT=PASS' } else { Write-Host ("SMOKE_VERDICT=FAIL ({0} check(s))" -f $fail) }
if ($fail -ne 0) { exit 1 }
exit 0
