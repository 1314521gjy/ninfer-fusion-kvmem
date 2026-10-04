# card-match-check.ps1 -- does this package belong on this GPU?
# (Chinese display name kept OUT of the file on purpose: this file must be ASCII-only, see below.)
#
# WHY: the package ships an sm_89 engine ONLY, and the engine carries no PTX for other
# architectures -- a wrong card does not fall back, it fails (or wedges). Two more real failures
# this guard exists for: (a) `nvidia-smi` missing / not reporting a compute capability, and
# (b) a card whose VRAM cannot hold weights + runtime (measured: 7.28 GiB + 1.54 GiB).
#
# ASCII-only on purpose (run by powershell.exe 5.1, which reads a BOM-less file as ANSI).
# -ForcedCc exists so the negative control can make THIS SCRIPT go red on a correct machine.
#
# Verdicts: HWCARD_* lines + HWCARD_VERDICT=PASS|FAIL ; exit 0/1.
param(
  [string]$Root = '',
  [string]$ForcedCc = '',
  [double]$MinimumVramGiB = 12
)
$ErrorActionPreference = 'Continue'
if (-not $Root) {
  # The kit root is the PARENT of this script's folder (this script ships in <kit>\engine\). The
  # previous default was Split-Path -Parent of the script itself, i.e. <kit>\engine, so every
  # relative lookup became engine\engine\... and the default invocation failed on a correct kit.
  $Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}
$Root = (Resolve-Path -LiteralPath $Root).Path

$failed = New-Object System.Collections.Generic.List[string]
function Verdict($name, $ok, $detail) {
  $state = if ($ok) { 'pass' } else { 'FAIL' }
  Write-Host ("HWCARD_{0}={1}{2}" -f $name, $state, $(if ($detail) { '  # ' + $detail } else { '' }))
  if (-not $ok) { [void]$failed.Add($name) }
}

Write-Host ('ROOT=' + $Root)

# ---- 1) engine present ----------------------------------------------------
# One binary per architecture. This check used to name a single file (ninfer-serve.exe) and accept a
# single capability (8.9), so it reported "does not match" for cards the kit now ships an engine for.
# It reports what is here, then compares that against the card.
$engineDir = Join-Path $Root 'engine'
$engines = @(Get-ChildItem -LiteralPath $engineDir -File -Filter 'ninfer-serve*.exe' -ErrorAction SilentlyContinue | Sort-Object Name)
if ($engines.Count -eq 0) {
  Verdict 'ENGINE' $false 'no ninfer-serve*.exe in engine\'
} else {
  Verdict 'ENGINE' $true (($engines | ForEach-Object { $_.Name + '=' + [math]::Round($_.Length / 1MB, 1) + 'MiB' }) -join ' ')
}
$wantByCc = @{ '8.6' = 'ninfer-serve-86.exe'; '8.9' = 'ninfer-serve-89.exe'; '12.0' = 'ninfer-serve-120a.exe' }

# ---- 2) nvidia-smi + compute capability ----------------------------------
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
  Verdict 'NVIDIA_SMI' $false 'nvidia-smi not found on PATH (install the NVIDIA driver)'
} else {
  Verdict 'NVIDIA_SMI' $true ('path=' + $smi.Source)
  $gpu = ''
  $cc = ''
  $vram = 0
  try {
    $line = (& nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($line) {
      $parts = $line -split ','
      if ($parts.Count -ge 3) {
        $gpu  = $parts[0].Trim()
        $cc   = $parts[1].Trim()
        $vram = [double](($parts[2] -replace '[^0-9\.]', '')) / 1024.0   # MiB -> GiB
      }
    }
  } catch { }
  if ($ForcedCc -ne '') { $cc = $ForcedCc; Write-Host ('HWCARD_CC_FORCED=' + $ForcedCc) }
  Write-Host ('HWCARD_GPU=' + $(if ($gpu) { $gpu } else { 'UNKNOWN' }))
  if ($cc -eq '') {
    Verdict 'CC' $false 'nvidia-smi did not report a compute capability'
  } else {
    $want = $wantByCc[$cc]
    if (-not $want) {
      Verdict 'CC' $false ('cc=' + $cc + ' -- this kit ships engines for 8.6 (RTX 30), 8.9 (RTX 40) and 12.0 (RTX 50) only')
    } elseif (-not (Test-Path -LiteralPath (Join-Path $engineDir $want))) {
      Verdict 'CC' $false ('cc=' + $cc + ' -- your card needs engine\' + $want + ', which is not in this package')
    } else {
      Verdict 'CC' $true ('cc=' + $cc + ' -> engine\' + $want)
    }
  }
  if ($vram -gt 0) {
    $vramOk = ($vram -ge $MinimumVramGiB)
    Verdict 'VRAM' $vramOk ('total_gib=' + [math]::Round($vram, 1) + ' required_gib=' + $MinimumVramGiB + $(if ($vramOk) { '' } else { ' -- too little VRAM for the tier you are about to load; see docs\11-*.md' }))
  } else {
    Verdict 'VRAM' $false 'could not read memory.total from nvidia-smi'
  }
}

# ---- 3) summary -----------------------------------------------------------
Write-Host ('HWCARD_FAILED=' + $failed.Count + $(if ($failed.Count) { '  :: ' + ($failed -join ',') } else { '' }))
Write-Host ('HWCARD_VERDICT=' + $(if ($failed.Count -eq 0) { 'PASS' } else { 'FAIL' }))
if ($failed.Count -eq 0) { exit 0 } else { exit 1 }
