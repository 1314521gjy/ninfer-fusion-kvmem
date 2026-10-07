# run-b-tier-batch.ps1 -- "B tier": the remaining RUNNABLE verifications, one pass.
#
# Scope (decided 2026-10-07; see the session goal):
#   B1  three-lever SPEED cost        base(full fat) vs all-three, same prompt, wall + TTFT from the
#                                     engine's own "req#N done ... TTFT" line. Decode rate is marked
#                                     DERIVED (=(output-1)/(wall-TTFT)) because neither log line gives
#                                     a per-request total.
#   B2  B06 multi-lane real run       scorer OFF + --max-concurrency 2 + ring OFF, two concurrent
#                                     requests with distinct markers -> each answer must carry its OWN
#                                     marker (the guard only proves the refusal; this proves the
#                                     non-scorer path still serves two lanes without cross-talk).
#   B3  kMin threshold dependence     reuse probe-kmin-sweep.ps1 with -Retrieve/-Share (two params
#                                     added to that wheel for this run): does the 4-page threshold move?
#
# REUSE POLICY: no new instrument. B1/B2 are driver loops around the engine plus its own self-reports;
# B3 calls the existing sweep. Anything else would be a wheel and must be registered instead.
# Pure ASCII on purpose (Windows PowerShell 5.1 reads BOM-less files as ANSI).

param(
  [string]$Exe   = 'E:\infer-build\p0-20261003\_verify-b01\build-ninja\apps\ninfer-serve.exe',
  [string]$Model = 'E:\ship-next\modelpack-ptq1-20261002\bonsai2_27b_ternary_ptq1_native_mtp.ninfer',
  [string]$OutRoot = 'E:\infer-build\p0-20261003\_verify-b01\b-tier',
  [string]$KminSweep = 'E:\infer-build\p0-20261003\probe-kmin-sweep.ps1',
  [int]$BasePort = 8700,
  [int]$ReadySec = 240,
  [int]$GenTokens = 128,
  [int]$Lanes = 2,
  [switch]$SkipB1, [switch]$SkipB2, [switch]$SkipB3
)
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$script:Port = $BasePort
$rows = New-Object System.Collections.ArrayList
$tsv  = Join-Path $OutRoot 'b-tier.tsv'
if (-not (Test-Path $Exe))   { Write-Host ("FATAL: exe not found: " + $Exe); exit 2 }
if (-not (Test-Path $Model)) { Write-Host ("FATAL: model not found: " + $Model); exit 2 }
$env:CUDA_PATH = 'E:\cuda-13.3'
$env:PATH = 'E:\cuda-13.3\bin;E:\cuda-13.3\bin\x64;' + (Split-Path $Exe -Parent) + ';' + $env:PATH
function Say([string]$t) { Write-Host ("[{0}] {1}" -f (Get-Date).ToString('HH:mm:ss'), $t) }
function Save { $rows | Export-Csv -Path $tsv -Delimiter "`t" -NoTypeInformation -Encoding UTF8 }

function Clear-Ring {
  foreach ($v in 'NINFER_KV_SINK','NINFER_KV_RING','NINFER_KV_WINDOW','NINFER_KV_RETRIEVE',
                 'NINFER_TERNARY_KVMEM','NINFER_TERNARY_KVMEM_SCORE','NINFER_HOST_PAGEABLE',
                 'NINFER_KV_REUSE_HOSTBACKED','NINFER_TERNARY_PTQ1_FAST','NINFER_KV_RETRIEVE_SHARE') {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
  }
}

function Wait-Port([int]$port) {
  for ($i = 0; $i -lt 15; $i++) {
    $own = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
           Select-Object -First 1 -ExpandProperty OwningProcess
    if (-not $own) { return }
    Stop-Process -Id $own -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2
  }
}

function Start-Engine {
  param([string]$Tag, [string[]]$Extra)
  $port = $script:Port; $script:Port = $script:Port + 1
  Wait-Port $port
  $dir = Join-Path $OutRoot $Tag
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $err = Join-Path $dir 'err.log'
  Remove-Item $err -ErrorAction SilentlyContinue
  $argv = @($Model,'--host','127.0.0.1','--port',"$port",'--model-id','qwen3.8-27b',
            '--kv-capacity','262144','--max-context','262144','--max-concurrency','1',
            '--no-thinking','--greedy','--host-kv-mib','16384') + $Extra
  $p = Start-Process -FilePath $Exe -ArgumentList $argv -PassThru -NoNewWindow `
         -WorkingDirectory (Split-Path $Exe -Parent) `
         -RedirectStandardOutput (Join-Path $dir 'out.log') -RedirectStandardError $err
  $ready = $false; $dl = (Get-Date).AddSeconds($ReadySec)
  while ((Get-Date) -lt $dl) {
    if ($p.HasExited) { break }
    try { $r = Invoke-WebRequest ("http://127.0.0.1:{0}/health" -f $port) -UseBasicParsing -TimeoutSec 3
          if ($r.StatusCode -eq 200) { $ready = $true; break } } catch { Start-Sleep -Milliseconds 600 }
  }
  return [pscustomobject]@{ tag=$Tag; port=$port; proc=$p; dir=$dir; err=$err; ready=$ready }
}

function Stop-Engine($e) {
  if ($e.proc -and -not $e.proc.HasExited) { Stop-Process -Id $e.proc.Id -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Milliseconds 800
  $own = Get-NetTCPConnection -LocalPort $e.port -State Listen -ErrorAction SilentlyContinue |
         Select-Object -First 1 -ExpandProperty OwningProcess
  if ($own) { Stop-Process -Id $own -Force -ErrorAction SilentlyContinue }
}

function Post-Chat([int]$port, [string]$prompt, [int]$maxTokens, [string]$outFile) {
  $body = @{ model='qwen3.8-27b'; messages=@(@{role='user';content=$prompt}); max_tokens=$maxTokens; stream=$false }
  $bf = "$outFile.body.json"
  [System.IO.File]::WriteAllText($bf, ($body | ConvertTo-Json -Depth 6 -Compress), (New-Object System.Text.UTF8Encoding($false)))
  $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $code = & $curl -s -o $outFile -w '%{http_code}' -X POST -H 'Content-Type: application/json' `
            --data-binary ("@" + $bf) ("http://127.0.0.1:{0}/v1/chat/completions" -f $port) 2>&1
  $sw.Stop()
  return [pscustomobject]@{ code = ($code -join '').Trim(); wall = [math]::Round($sw.Elapsed.TotalSeconds,2) }
}

function Log-Stats([string]$errLog) {
  # MEASURED 2026-10-07 (B1 rerun): the first version grabbed a BARE number after "TTFT" and wrote it
  # into a column named ttft_s, while the engine prints "TTFT 367 ms" -- so the row said TTFT=367 s
  # next to wall=0.51 s, and the derived decode rate (output-1)/(wall-TTFT) came out negative. The
  # engine ALREADY self-reports TTFT / total / prefill / decode, so the honest fix is to read those
  # fields WITH their units and to stop calling decode an exported quantity.
  $ttft=''; $total=''; $prefill=''; $decode=''; $prompt=''; $outN=''; $pool=''; $ready=0
  if (Test-Path $errLog) {
    $pl = (Select-String -Path $errLog -Pattern 'capacity \| KV' -EA SilentlyContinue | Select-Object -First 1).Line
    if ($pl) { $pool = $pl.Trim() }
    $rl = (Select-String -Path $errLog -Pattern 'req#\d+ done' -EA SilentlyContinue | Select-Object -Last 1).Line
    if ($rl) {
      $m = [regex]::Match($rl, 'prompt (\d+)');          if ($m.Success) { $prompt  = $m.Groups[1].Value }
      $m = [regex]::Match($rl, 'output (\d+)');          if ($m.Success) { $outN    = $m.Groups[1].Value }
      $m = [regex]::Match($rl, 'TTFT ([\d.]+) ms');      if ($m.Success) { $ttft    = $m.Groups[1].Value }
      $m = [regex]::Match($rl, 'total ([\d.]+) ms');     if ($m.Success) { $total   = $m.Groups[1].Value }
      $m = [regex]::Match($rl, 'prefill ([\d.]+)k tok/s'); if ($m.Success) { $prefill = $m.Groups[1].Value }
      $m = [regex]::Match($rl, 'decode ([\d.]+) tok/s'); if ($m.Success) { $decode  = $m.Groups[1].Value }
    }
    $ready = ([regex]::Matches((Get-Content -Raw $errLog), 'engine ready')).Count
  }
  return [pscustomobject]@{ pool=$pool; prompt=$prompt; out=$outN; ttft_ms=$ttft; total_ms=$total
                            prefill=$prefill; decode=$decode; ready=$ready }
}

Say "=== B-tier batch START ==="
Say ("exe    = " + $Exe)
Say ("sha256 = " + (Get-FileHash $Exe -Algorithm SHA256).Hash)
Say ("stray  = " + @(Get-Process -Name ninfer,ninfer-serve -EA SilentlyContinue).Count)

# ---------------------------------------------------------------- B1: speed cost of the three levers
if (-not $SkipB1) {
  Say ""
  Say "### B1 speed cost: base vs all-three (same prompt, same binary) ###"
  $prompt = ('Summarise the following maintenance log in one sentence: ' + ('the archive ledger records mundane line ' * 120))
  foreach ($cfg in @(
      @{ tag='b1-base';      extra=@('--prefill-chunk','1024') },
      @{ tag='b1-all-three'; extra=@('--gdn-state-fp16','--prefill-chunk','256','--no-cuda-graph') })) {
    Clear-Ring
    $pristine = $false
    try {
      $e = Start-Engine -Tag $cfg.tag -Extra $cfg.extra
      $pristine = $e.ready
      $http=''; $wall=''
      if ($e.ready) {
        $res = Post-Chat -port $e.port -prompt $prompt -maxTokens $GenTokens -outFile (Join-Path $e.dir 'resp.json')
        $http = $res.code; $wall = $res.wall
        Start-Sleep -Seconds 2
      }
      $st = Log-Stats $e.err
      $derived = ''
      if ($st.out -and $st.ttft_ms -and $wall) {
        $d = [double]$wall - ([double]$st.ttft_ms / 1000.0)
        if ($d -gt 0.001) { $derived = [math]::Round((([double]$st.out) - 1.0) / $d, 2) }
      }
      $row = [pscustomobject]@{ section='B1'; name=$cfg.tag; extra=($cfg.extra -join ' '); ready=[int]$pristine
        http=$http; wall_s=$wall; prompt_tokens=$st.prompt; output_tokens=$st.out
        ttft_ms=$st.ttft_ms; total_ms=$st.total_ms; prefill_tok_s=$st.prefill; decode_tok_s=$st.decode
        decode_tok_s_DERIVED=$derived; pool=$st.pool
        note='TTFT/total/prefill/decode are the ENGINE self-report (ms, ms, k tok/s, tok/s); the DERIVED column is only a cross-check and is empty when wall < TTFT/1000' }
      $rows.Add($row) | Out-Null; Save
      Say ("  {0,-14} ready={1} http={2} wall={3}s prompt={4} out={5} TTFT={6}ms total={7}ms prefill={8}k decode={9} tok/s (derived={10})" -f `
           $cfg.tag,$pristine,$http,$wall,$st.prompt,$st.out,$st.ttft_ms,$st.total_ms,$st.prefill,$st.decode,$derived)
    } finally { if ($e) { Stop-Engine $e } }
  }
}

# ---------------------------------------------------------------- B2: N lanes, scorer OFF
if (-not $SkipB2) {
  Say ""
  Say ("### B2 " + $Lanes + " concurrent lanes with the scorer OFF (KVMEM=0) ###")
  Clear-Ring
  $env:NINFER_TERNARY_KVMEM = '0'
  $e = $null
  $tag = if ($Lanes -eq 2) { 'b2-two-lanes' } else { 'b2-' + $Lanes + 'lanes-kvmem0' }
  try {
    $e = Start-Engine -Tag $tag -Extra @('--max-concurrency', "$Lanes")
    if ($e.ready) {
      $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
      # MEASURED 2026-10-07 (first run of this batch): Start-Process -ArgumentList MANGLES
      # `-w %{http_code}` and `--data-binary @file`, so the response body landed in the .code file and
      # the parsed row came out garbage (http="000{json}200", own_marker=False) while the ENGINE log
      # showed a clean two-lane run. Each lane therefore runs in its own Start-Job and calls curl
      # through the call operator, which keeps the argument vector intact.
      # ADDED 2026-10-07 (closes the internal item #18): -Lanes N with distinct markers, plus a
      # POST-workload health request -- /v1/models stays 200 even with a dead worker, so survival has
      # to be proved with a fresh completion, not with a liveness endpoint.
      $words = @('ALPHA','BETA','GAMMA','DELTA','EPSILON','ZETA','ETA','THETA')
      $laneDefs = @()
      for ($i = 0; $i -lt $Lanes; $i++) {
        $laneDefs += [pscustomobject]@{ n = ("lane" + ($i + 1)); w = $words[$i % $words.Count] }
      }
      $jobs = @()
      foreach ($lane in $laneDefs) {
        $b = @{ model='qwen3.8-27b'; messages=@(@{ role='user'; content=('Reply with exactly this word: ' + $lane.w) }); max_tokens=24; stream=$false }
        $bf = Join-Path $e.dir ($lane.n + '.body.json')
        [System.IO.File]::WriteAllText($bf, ($b | ConvertTo-Json -Depth 6 -Compress), (New-Object System.Text.UTF8Encoding($false)))
        $outF = Join-Path $e.dir ($lane.n + '.json')
        $url  = ("http://127.0.0.1:{0}/v1/chat/completions" -f $e.port)
        $j = Start-Job -ScriptBlock {
          param($curl, $out, $body, $u)
          & $curl -s -o $out -w '%{http_code}' -X POST -H 'Content-Type: application/json' --data-binary ("@" + $body) $u
        } -ArgumentList $curl, $outF, $bf, $url
        $jobs += [pscustomobject]@{ n = $lane.n; job = $j }
      }
      $jobs | ForEach-Object { $_.job | Wait-Job -Timeout 300 | Out-Null }
      $code = @{}
      foreach ($x in $jobs) { $code[$x.n] = (@(Receive-Job $x.job) -join '').Trim(); Remove-Job $x.job -Force }
      Start-Sleep -Seconds 1
      $okAll = $true; $codes = @(); $own = @()
      foreach ($lane in $laneDefs) {
        $body = [string](Get-Content (Join-Path $e.dir ($lane.n + '.json')) -Raw -EA SilentlyContinue)
        $hitOwn = ($body -match $lane.w)
        $hitOther = $false
        foreach ($other in @($laneDefs | Where-Object { $_.n -ne $lane.n })) { if ($body -match $other.w) { $hitOther = $true } }
        $ok = $hitOwn -and (-not $hitOther)
        if (-not $ok) { $okAll = $false }
        $codes += [string]$code[$lane.n]
        $own += ($lane.n + "=" + $ok)
      }
      $hb = @{ model='qwen3.8-27b'; messages=@(@{ role='user'; content='Reply with exactly this word: OMEGA' }); max_tokens=8; stream=$false }
      $hbf = Join-Path $e.dir 'after.body.json'
      [System.IO.File]::WriteAllText($hbf, ($hb | ConvertTo-Json -Depth 6 -Compress), (New-Object System.Text.UTF8Encoding($false)))
      $afterRaw = & $curl -s -o (Join-Path $e.dir 'after.json') -w '%{http_code}' -X POST -H 'Content-Type: application/json' --data-binary ("@" + $hbf) ("http://127.0.0.1:{0}/v1/chat/completions" -f $e.port) 2>&1
      $afterCode = ($afterRaw -join '').Trim()
      $afterBody = [string](Get-Content (Join-Path $e.dir 'after.json') -Raw -EA SilentlyContinue)
      $alive = ($afterCode -eq '200') -and ($afterBody -match 'OMEGA')
      $st = Log-Stats $e.err
      $row = [pscustomobject]@{ section='B2'; name=$tag; extra=("--max-concurrency " + $Lanes); ready=[int]$e.ready
        http=($codes -join '/'); wall_s=''; ttft_ms=$st.ttft_ms; output_tokens=''; decode_tok_s_DERIVED=''; pool=$st.pool
        note=("lanes=" + $Lanes + " own_marker[" + ($own -join ' ') + "] no_crosstalk=" + $okAll +
              " after_health_http=" + $afterCode + " engine_alive=" + $alive +
              " ; ttft_ms is the engine self-report of the LAST lane") }
      $rows.Add($row) | Out-Null; Save
      Say ("  {0} lanes: http={1}  own_marker[{2}]  no cross-talk={3}  after_health={4} alive={5}" -f $Lanes,($codes -join '/'),($own -join ' '),$okAll,$afterCode,$alive)
      foreach ($lane in $laneDefs) {
        $body = [string](Get-Content (Join-Path $e.dir ($lane.n + '.json')) -Raw -EA SilentlyContinue)
        Say ("    " + $lane.n + " body (first 120): " + $body.Substring(0,[math]::Min(120,$body.Length)))
      }
    } else {
      # MEASURED 2026-10-07 (B2 rerun): the old one-liner Say "..." + $e.err dropped the path, so the
      # failure printed an empty location and cost a whole diagnostic round. Print the path AND the
      # tail ERROR lines, so a dead engine is self-explaining at the point of failure.
      Say ("  engine did not start; see " + $e.err)
      if (Test-Path $e.err) {
        Get-Content $e.err -Tail 14 -EA SilentlyContinue |
          Where-Object { $_ -match 'ERROR|FATAL|failed|abort' } |
          ForEach-Object { Say ("    | " + $_.Trim()) }
      }
    }
  } finally { if ($e) { Stop-Engine $e }; Remove-Item Env:NINFER_TERNARY_KVMEM -EA SilentlyContinue }
}

# ---------------------------------------------------------------- B3: kMin threshold dependence
if (-not $SkipB3) {
  Say ""
  Say "### B3 does the 4-page kMin threshold depend on NINFER_KV_RETRIEVE / SHARE? ###"
  foreach ($cfg in @(
      @{ tag='ret768-share50';  retrieve=768;  share=50 },
      @{ tag='ret1536-share50'; retrieve=1536; share=50 },
      @{ tag='ret768-share25';  retrieve=768;  share=25 })) {
    Say ("  --- " + $cfg.tag + " (RETRIEVE=" + $cfg.retrieve + " SHARE=" + $cfg.share + ") ---")
    & $KminSweep -Exe $Exe -Model $Model -Pools @(59,60,61) -SinkTokens 2816 `
        -Retrieve $cfg.retrieve -Share $cfg.share -Root (Join-Path $OutRoot ("kmin-" + $cfg.tag))
    $t = Join-Path $OutRoot ("kmin-" + $cfg.tag + "\kmin-sweep.tsv")
    if (Test-Path $t) {
      Import-Csv $t -Delimiter "`t" | ForEach-Object {
        $row = [pscustomobject]@{ section='B3'; name=($cfg.tag + '-p' + $_.pool); extra=("RETRIEVE=" + $cfg.retrieve + " SHARE=" + $cfg.share)
          ready=1; http=$_.http; wall_s=''; ttft_s=''; output_tokens=''; decode_tok_s_DERIVED=''; pool=("pool=" + $_.pool + " evidence=" + $_.evidence + " free=" + $_.free)
          note=("tool=" + $_.tool) }
        $script:rows.Add($row) | Out-Null
      }
      Save
    }
  }
}

Say ""
Say "=== B-tier batch END ==="
Say ("rows = " + $rows.Count + "   tsv = " + $tsv)
Say ("stray = " + @(Get-Process -Name ninfer,ninfer-serve -EA SilentlyContinue).Count)
Say ("gpu = " + ((& nvidia-smi --query-gpu=memory.used --format=csv,noheader) -join ''))
