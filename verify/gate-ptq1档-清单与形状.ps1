# kitcheck-ptq1.ps1 -- PTQ1_0 beta-kit self-consistency gate.
#
# WHY: "a human looked at it once" is not proof. This gate re-checks, in one shot, the things that
# have actually broken on this project for THIS kit's shape:
#   * every file listed in <kit>\SHA256SUMS.txt hashes as recorded   (frozen-copy integrity)
#   * the three architecture engines are present and non-empty       (one cubin per arch, no fallback)
#   * the engine's runtime DLLs sit NEXT TO the exe                  (else 0xC0000135 at startup)
#   * the launcher is ASCII-only                                     (cmd parses .bat in the OEM CP)
#   * the launcher carries the five ring switches + --max-shared-prefixes 0
#   * **the launcher carries NINFER_TERNARY_PTQ1_FAST=1**            (without it PTQ1_0 prefill
#     measures ~20x slower with no error line: landing table section 28.46)
#   * the PTQ1_0 model is present with the recorded byte count
#
# ASCII-ONLY on purpose: powershell.exe 5.1 reads a BOM-less .ps1 as ANSI, so a non-ASCII literal here
# would be silently mangled. Non-ASCII file NAME is handled by extension/pattern, never by a literal.
#
# Exit: 0 when everything passes, 1 otherwise. Every check prints one greppable line.
# usage: powershell -File <kit>\engine\kitcheck-ptq1.ps1 [-Kit <kit>] [-NoModels]

param([string]$Kit = (Split-Path -Parent $PSScriptRoot), [switch]$NoModels)

$ErrorActionPreference = 'Continue'
$fail = 0
function Check([string]$Name, [bool]$Ok, [string]$Detail) {
  $verdict = 'OK  '
  if (-not $Ok) { $verdict = 'FAIL'; $script:fail++ }
  Write-Host ("[{0}] {1,-46} {2}" -f $verdict, $Name, $Detail)
}

Write-Host ("KIT = {0}" -f $Kit)

# ---- 1. path rules: the native exe cannot read a non-ASCII path --------------------------------
function Is-Ascii([string]$s) { foreach ($ch in $s.ToCharArray()) { if ([int]$ch -gt 127) { return $false } } return $true }
Check 'kit path is ASCII' (Is-Ascii $Kit) $Kit
$modelsDir = Join-Path $Kit 'models'
$modelFiles = @(Get-ChildItem -LiteralPath $modelsDir -File -Filter '*.ninfer' -ErrorAction SilentlyContinue | Sort-Object Name)
$badName = @($modelFiles | Where-Object { -not (Is-Ascii $_.Name) })
Check 'model file names are ASCII' ($badName.Count -eq 0) ("{0} file(s), {1} non-ASCII" -f $modelFiles.Count, $badName.Count)

# ---- 2. the model (NOT shipped with this kit since 2026-10-02) ----------------------------------
# The weights travel as a SEPARATE model pack, so a missing model file is not a failure here: the
# kit is complete without it. What IS a failure: a model that sits in models\ with the WRONG byte
# count -- a truncated or renamed download, i.e. exactly what this check can still catch.
# The expected artifact: PTQ1_0 native + MTP head, 6,394,697,216 B (measured on the shipped copy AND
# on the source artifact; the first draft of this gate carried a wrong constant and reported the model
# as broken -- a gate that checks the wrong number is worse than no gate).
# -NoModels stays accepted as a no-op switch so older instructions keep working.
$expected = @{ 'bonsai2_27b_ternary_ptq1_native_mtp.ninfer' = 6394697216 }
$present = 0
foreach ($name in $expected.Keys) {
  $f = Get-Item -LiteralPath (Join-Path $modelsDir $name) -ErrorAction SilentlyContinue
  if ($null -eq $f) { continue }
  $present++
  Check ("model present, byte count: {0}" -f $name) ($f.Length -eq $expected[$name]) ("expected {0:N0} B, got {1:N0} B" -f $expected[$name], $f.Length)
}
Check 'model is NOT shipped in this kit (separate model pack)' $true ("{0} of {1} downloaded; a missing model is not a failure" -f $present, $expected.Count)

# ---- 3. the three engines and their runtime DLLs -----------------------------------------------
$engines = @('ninfer-serve-86.exe','ninfer-serve-89.exe','ninfer-serve-120a.exe')
$engDir = Join-Path $Kit 'engine'
foreach ($e in $engines) {
  $f = Get-Item -LiteralPath (Join-Path $engDir $e) -ErrorAction SilentlyContinue
  Check ("engine present: {0}" -f $e) (($null -ne $f) -and ($f.Length -gt 100MB)) $(if ($f) { "{0:N1} MB" -f ($f.Length/1MB) } else { 'missing' })
}
$needDlls = @('cudart64_13.dll','cublas64_13.dll','cublasLt64_13.dll','avcodec-63.dll','avformat-63.dll','avutil-61.dll','swresample-7.dll','swscale-10.dll','libcurl-x64.dll')
$missing = @($needDlls | Where-Object { -not (Test-Path -LiteralPath (Join-Path $engDir $_)) })
Check 'runtime DLLs sit next to the exe' ($missing.Count -eq 0) ("{0}/{1} present{2}" -f ($needDlls.Count - $missing.Count), $needDlls.Count, $(if ($missing.Count) { "; missing " + ($missing -join ',') } else { '' }))
Check 'engine picker present' (Test-Path -LiteralPath (Join-Path $engDir 'pick-engine.bat')) 'pick-engine.bat'

# ---- 4. the launcher --------------------------------------------------------------------------
$bats = @(Get-ChildItem -LiteralPath $Kit -File -Filter 'start-*.bat')
Check 'at least one launcher' ($bats.Count -gt 0) (($bats | ForEach-Object { $_.Name }) -join ', ')
foreach ($b in $bats) {
  $bytes = [IO.File]::ReadAllBytes($b.FullName)
  $nonAscii = @($bytes | Where-Object { $_ -gt 127 }).Count
  Check ("launcher ASCII-only: {0}" -f $b.Name) ($nonAscii -eq 0) ("{0} non-ASCII byte(s)" -f $nonAscii)
  $t = [Text.Encoding]::ASCII.GetString($bytes)
  foreach ($need in @('NINFER_KV_WINDOW','NINFER_KV_RETRIEVE','NINFER_KV_RING','NINFER_HOST_PAGEABLE','NINFER_KV_REUSE_HOSTBACKED','--max-shared-prefixes 0')) {
    Check ("  {0} carries {1}" -f $b.Name, $need) ($t -match [regex]::Escape($need)) ''
  }
  if ($b.Name -like '*ptq1*') {
    Check ("  {0} carries NINFER_TERNARY_PTQ1_FAST=1" -f $b.Name) ($t -match 'NINFER_TERNARY_PTQ1_FAST=1') 'the load-bearing switch (landing table section 28.46)'
  }
}

# ---- 5. frozen-copy integrity -----------------------------------------------------------------
$sums = Join-Path $Kit 'SHA256SUMS.txt'
if (-not (Test-Path -LiteralPath $sums)) {
  Check 'SHA256SUMS.txt present' $false 'run build-ptq1-kit.ps1 to produce it'
} else {
  # READ AS UTF-8, ALWAYS: the hashed tree contains tool scripts with non-ASCII names, and
  # Get-Content on a BOM-less file uses the ANSI codepage in powershell 5.1 -- which mangles exactly
  # those names, makes Test-Path fail, and reports them as "missing" (measured: 5 of 21 entries).
  $lines = @([IO.File]::ReadAllLines($sums, [Text.Encoding]::UTF8) | Where-Object { $_ -match '^[0-9A-Fa-f]{64}  ' })
  $badHash = 0; $badMissing = 0
  foreach ($l in $lines) {
    $h = $l.Substring(0,64); $rel = $l.Substring(66)
    $p = Join-Path $Kit $rel
    if (-not (Test-Path -LiteralPath $p)) { $badMissing++; continue }
    if ((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash -ne $h) { $badHash++ }
  }
  Check 'every hashed file matches SHA256SUMS.txt' (($badHash -eq 0) -and ($badMissing -eq 0)) ("{0} entries; {1} mismatched, {2} missing" -f $lines.Count, $badHash, $badMissing)
  # .ninfer is excluded: models are NOT part of this kit (they travel as a separate model pack), so a
  # downloaded model dropped into models\ must not read as "an unhashed file in the kit".
  $onDisk = @(Get-ChildItem -LiteralPath $Kit -Recurse -File | Where-Object { $_.Name -notin @('SHA256SUMS.txt','KIT.txt') -and $_.Extension -ne '.ninfer' }).Count
  Check 'no unhashed file in the kit' ($onDisk -eq $lines.Count) ("{0} files on disk vs {1} hashed" -f $onDisk, $lines.Count)
}

Write-Host ''
Write-Host ("KITCHECK_VERDICT={0}  ({1} failure(s))" -f $(if ($fail -eq 0) { 'PASS' } else { 'FAIL' }), $fail)
exit $(if ($fail -eq 0) { 0 } else { 1 })
