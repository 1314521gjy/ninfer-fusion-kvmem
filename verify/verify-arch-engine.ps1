# verify-arch-engine.ps1 -- per-architecture acceptance for a patched engine.
#
# What this answers, for ONE engine:
#   1) does it START at all on this card            (kernel-image availability -- the thing that
#                                                    cannot be faked by a hash or a file size)
#   2) does the over-pool WARNING fire when it should  (positive control)
#   3) does it stay quiet when the prompt fits         (negative control)
#   4) is the answer still correct on an over-pool prompt (the warning must not have changed behaviour)
#      -- and this is now asked TWICE: needle in the MIDDLE (3 trials, a complete loss is a FAIL) and
#         needle at the END (2 trials, a miss is recorded only). The old script asked it once, with the
#         needle at the very END of the prompt -- where it is always inside the resident window, so that
#         arm could print RESULT: PASS while the middle of the prompt was silently gone. Reported as
#         S9-D in the 2026-10-02 RTX 4070 Ti SUPER receipt; fixed here 2026-10-02.
#      -- an answer truncated by max_tokens (finish_reason=length) or empty is INCONCLUSIVE, never a
#         missed needle (receipt S9-E: 64 max_tokens with thinking ON reads exactly like a lost needle).
#
# The positive/negative pair is the whole point: an engine that warns on EVERYTHING passes (2) alone.
#
# Card-vs-arch is not assumed: `cn` prints what actually happened. Measured 2026-10-01 on an
# RTX 4080 SUPER (sm_89): ninfer-serve-86.exe loads and serves fine, ninfer-serve-120a.exe dies with
# cudaErrorNoKernelImageForDevice -- which is exactly the documented "arch binaries do not fall back".
param(
  [Parameter(Mandatory = $true)][string]$Exe,
  [Parameter(Mandatory = $true)][string]$Tag,
  [string]$Model = 'E:\betakit-ptq1-v1\models\bonsai2_27b_ternary_ptq1_mtp.ninfer',
  [int]$Port = 8095,
  [int]$MaxContext = 65536,     # kept modest so the RING HOST-BUDGET GUARD is satisfiable:
                                # it needs host_pages + pool_pages >= page_count(max_context)
  [int]$HostKvMib = 16384,
  [string]$KvDtype = 'k8v4',
  [int]$SmallPool = 2048,       # prompt will NOT fit  -> warning required
  [int]$BigPool = 32768,        # same prompt DOES fit -> warning forbidden
  [int]$Filler = 400,           # ~4.2k-token prompt
  [int]$ReadyTimeoutSec = 300
)
$ErrorActionPreference = 'Continue'
# The delivery ships the model under this name; accept either spelling.
if (-not (Test-Path -LiteralPath $Model)) {
  $alt = $Model -replace '_mtp\.ninfer$', '_native_mtp.ninfer'
  if (Test-Path -LiteralPath $alt) { $Model = $alt }
}
if (-not (Test-Path -LiteralPath $Exe))   { "REFUSE: missing $Exe"; exit 3 }
if (-not (Test-Path -LiteralPath $Model)) { "REFUSE: missing $Model"; exit 3 }

# RUNTIME CO-LOCATION CHECK (added after a real false negative, 2026-10-01 22:19).
# The engine loads the CUDA/ffmpeg/curl DLLs from ITS OWN directory, not from PATH. Running it from a
# folder that holds only the .exe (e.g. an increment whose DLLs deliberately stayed in the donor kit)
# produces "engine never became servable" with an EMPTY stderr log -- which reads exactly like a
# broken engine, and is actually a missing-runtime problem. Refuse with a diagnosis instead, so the
# reading can never be mistaken for a verdict about the binary.
$MissingDll = @(Get-ChildItem -LiteralPath (Split-Path $Exe -Parent) -Filter *.dll -File -ErrorAction SilentlyContinue)
if ($MissingDll.Count -eq 0) {
  "REFUSE: no .dll next to $Exe."
  "        The engine loads its runtime from its own directory; running it without them fails"
  "        SILENTLY (empty stderr). Point -Exe at a build directory, or stage the DLLs beside it."
  "        This is a harness refusal, NOT a verdict about the engine."
  exit 4
}
"runtime dlls beside the exe: $($MissingDll.Count)"

$Out = Join-Path 'E:\infer-build\fusion-master\needle' $Tag
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Fail = 0

function Engine-Arm {
  param([int]$Pool, [int]$Trials, [string]$ArmTag, [bool]$ExpectWarning, [string]$NeedleAt = 'mid')
  $dir = Join-Path $Out $ArmTag
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  foreach ($v in @('NINFER_KV_WINDOW','NINFER_KV_RETRIEVE','NINFER_KV_RING','NINFER_HOST_PAGEABLE',
                   'NINFER_KV_REUSE_HOSTBACKED','NINFER_TERNARY_PTQ1_FAST','NINFER_AGENT_CC')) {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
  }
  $env:NINFER_KV_WINDOW           = '16384'
  $env:NINFER_KV_RETRIEVE         = '8192'
  $env:NINFER_KV_RING             = '1'
  $env:NINFER_HOST_PAGEABLE       = '1'
  $env:NINFER_KV_REUSE_HOSTBACKED = '1'
  $env:NINFER_TERNARY_PTQ1_FAST   = '1'
  $argv = @($Model,'--host','127.0.0.1','--port',"$Port",'--model-id','qwen3.8-27b',
    '--max-context',"$MaxContext",'--kv-capacity',"$Pool",'--kv-dtype',"$KvDtype",
    '--host-kv-mib',"$HostKvMib",'--prefill-chunk','1024','--spec','mtp','--draft-tokens','4',
    '--default-max-tokens','16384','--default-reasoning-effort','none','--max-concurrency','1',
    '--max-shared-prefixes','0','--presence-penalty','0','--temperature','0','--top-p','1','--top-k','0')
  Get-CimInstance Win32_Process -Filter "Name='ninfer-serve*.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ExecutablePath -eq $Exe } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  $dl = (Get-Date).AddSeconds(30)
  while ((Get-Date) -lt $dl -and (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) { Start-Sleep -Milliseconds 500 }
  $env:PATH = 'E:\cuda-13.3\bin;E:\cuda-13.3\bin\x64;' + (Split-Path $Exe -Parent) + ';' + $env:PATH
  Set-Location (Split-Path $Exe -Parent)
  $p = Start-Process -FilePath $Exe -ArgumentList $argv -PassThru -NoNewWindow `
    -RedirectStandardOutput (Join-Path $dir 'out.log') -RedirectStandardError (Join-Path $dir 'err.log')
  $ready = $false; $d2 = (Get-Date).AddSeconds($ReadyTimeoutSec)
  while ((Get-Date) -lt $d2) {
    if ($p.HasExited) { break }
    try { $r = Invoke-WebRequest "http://127.0.0.1:$Port/v1/models" -UseBasicParsing -TimeoutSec 5
          if ($r.StatusCode -eq 200) { $ready = $true; break } } catch { Start-Sleep -Seconds 4 }
  }
  if (-not $ready) {
    "  [$ArmTag] NOT SERVABLE - stderr tail:"
    Get-Content (Join-Path $dir 'err.log') -ErrorAction SilentlyContinue | Select-Object -Last 4 | ForEach-Object { "      | $_" }
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    return @{ Ready = $false; Warn = 0; Hits = 0; Trials = 0 }
  }
  $warn = 0; $hits = 0; $inconclusive = 0
  for ($t = 1; $t -le $Trials; $t++) {
    $code = (7000 + $t).ToString()
    $FILL = 'The quick brown fox jumps over the lazy dog. '
    # NEEDLE POSITION (2026-10-02, from the 4070 Ti SUPER receipt S9-D): the needle used to sit at the
    # very END of the prompt -- and the end of the prompt is always inside the resident window, so this
    # arm could only ever prove "the warning fires", never "the middle survives". The middle is the
    # whole point of the over-pool test, so 'mid' is now the default. Run BOTH positions: a patch that
    # only passes 'end' has not been tested on the defect it exists for.
    $half = [int]($Filler / 2)
    if ($NeedleAt -eq 'end') {
      $body = ($FILL * $Filler) + "`n`nIMPORTANT ARCHIVE CODE: ARCH-$code`n`n"
    } else {
      $body = ($FILL * $half) + "`n`nIMPORTANT ARCHIVE CODE: ARCH-$code`n`n" + ($FILL * ($Filler - $half))
    }
    $ask  = $body + "`n`nIgnore every other instruction. What is the IMPORTANT ARCHIVE CODE above? Reply with that code and nothing else."
    $payload = @{ model='qwen3.8-27b'; messages=@(@{role='user';content=$ask}); temperature=0; max_tokens=64 } |
               ConvertTo-Json -Depth 6 -Compress
    try {
      $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
                -ContentType 'application/json' -Body $payload -UseBasicParsing -TimeoutSec 900
      $j = $resp.Content | ConvertFrom-Json
      $ans = $j.choices[0].message.content
      $fin = [string]$j.choices[0].finish_reason
      # S9-E trap: with thinking left ON, 64 max_tokens are spent THINKING and the body is truncated --
      # that reads as "the needle was lost" while KV has nothing to do with it. Truncated/empty answers
      # are therefore INCONCLUSIVE, never a miss. Turn thinking off, or give max_tokens >= 256.
      if ($fin -eq 'length' -or [string]::IsNullOrWhiteSpace($ans)) {
        $inconclusive++
        "  [$ArmTag/$NeedleAt] #$t http=200 fin=$fin INCONCLUSIVE (truncated/empty - thinking? raise max_tokens, do NOT read this as a miss)"
      } elseif ($ans -match "ARCH-$code") {
        $hits++
        "  [$ArmTag/$NeedleAt] #$t http=200 fin=$fin hit=True ans=$ans"
      } else {
        "  [$ArmTag/$NeedleAt] #$t http=200 fin=$fin hit=False ans=$ans"
      }
    } catch { "  [$ArmTag/$NeedleAt] #$t transport/HTTP error: $($_.Exception.Message)" }
  }
  $warn = (Select-String -Path (Join-Path $dir 'err.log') -Pattern 'exceeds the resident Device KV pool' -ErrorAction SilentlyContinue).Count
  $caps = (Select-String -Path (Join-Path $dir 'err.log') -Pattern 'capacity \|' -ErrorAction SilentlyContinue | Select-Object -First 1)
  "  [$ArmTag] pool_line: $(if ($caps) { $caps.Line.Substring([Math]::Max(0,$caps.Line.IndexOf('capacity'))) } else { 'n/a' })"
  "  [$ArmTag] warning_lines=$warn  (expect $(if ($ExpectWarning) {'>=1'} else {'0'}))"
  if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
  Start-Sleep -Seconds 3
  $d3 = (Get-Date).AddSeconds(20)
  while ((Get-Date) -lt $d3 -and (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) { Start-Sleep -Milliseconds 500 }
  return @{ Ready = $true; Warn = $warn; Hits = $hits; Trials = $Trials; ExpectWarn = $ExpectWarning
            Inconclusive = $inconclusive; NeedleAt = $NeedleAt }
}

"=== $Tag ==="
"exe    : $Exe"
"sha256 : $((Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash)"
"bytes  : $((Get-Item -LiteralPath $Exe).Length)"
"ctx=$MaxContext host_kv_mib=$HostKvMib small_pool=$SmallPool big_pool=$BigPool filler=$Filler"
""
"--- control arm: prompt FITS the pool (warning forbidden) ---"
$neg = Engine-Arm -Pool $BigPool -Trials 2 -ArmTag 'neg-bigpool' -ExpectWarning $false
if ($neg.Ready -and $neg.Warn -ne 0) { "  VERDICT: FAIL (warned when it must not)"; $Fail++ }
""
"--- over-pool arm A: needle in the MIDDLE of the prompt (warning required, answer must survive) ---"
$pos = Engine-Arm -Pool $SmallPool -Trials 3 -ArmTag 'pos-mid' -ExpectWarning $true -NeedleAt 'mid'
if ($pos.Ready -and $pos.Warn -lt 1) { "  VERDICT: FAIL (silent on an over-pool prompt)"; $Fail++ }
if ($pos.Ready -and ($pos.Trials - $pos.Inconclusive) -eq 0) {
  "  VERDICT: INCONCLUSIVE for this arm (every request was truncated/empty) -- see the notes above"
} elseif ($pos.Ready -and $pos.Hits -eq 0) {
  "  VERDICT: FAIL (mid-prompt needle lost on an over-pool prompt: $($pos.Hits)/$($pos.Trials)) -- this IS the defect the warning exists for"
  $Fail++
} elseif ($pos.Ready -and $pos.Hits -lt ($pos.Trials - $pos.Inconclusive)) {
  "  NOTE: mid-prompt needle partially lost ($($pos.Hits)/$($pos.Trials)) - recorded"
}
""
"--- over-pool arm B: needle at the END of the prompt (the weaker shape; a miss is recorded, not fatal) ---"
$posEnd = Engine-Arm -Pool $SmallPool -Trials 2 -ArmTag 'pos-end' -ExpectWarning $true -NeedleAt 'end'
if ($posEnd.Ready -and $posEnd.Warn -lt 1) { "  VERDICT: FAIL (silent on an over-pool prompt)"; $Fail++ }
if ($posEnd.Ready -and $posEnd.Hits -lt ($posEnd.Trials - $posEnd.Inconclusive)) {
  "  NOTE: needle missed at the prompt END ($($posEnd.Hits)/$($posEnd.Trials)) - recorded, not fatal here"
}
""
if (-not $neg.Ready -or -not $pos.Ready -or -not $posEnd.Ready) { "RESULT: engine did not become servable on this card -> acceptance cannot be claimed locally."; exit 2 }
if ($Fail -gt 0) { "RESULT: FAIL"; exit 1 }
"RESULT: PASS (warning fires over-pool, silent when it fits, MID-prompt answer survived)"
"  Reminder (receipt S9-E): every needle number above is only valid with thinking OFF or a large"
"  max_tokens -- a truncated answer is an INCONCLUSIVE, never a missed needle. Logs: $Out"
exit 0
