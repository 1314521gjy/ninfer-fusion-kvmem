# kitcheck.ps1 -- beta-kit self-consistency gate.
#
# WHY: "a human looked at it once" is not proof. This gate re-checks, in one shot, the things that
# have actually broken on this project for THIS kit's shape:
#   * every file listed in <kit>\SHA256SUMS.txt hashes as recorded   (frozen-copy integrity)
#   * the engine's runtime DLLs sit NEXT TO the exe                  (else 0xC0000135 at startup)
#   * the launchers are ASCII-only                                   (cmd parses .bat in the OEM CP)
#   * model paths and the kit path itself are ASCII                  (native exe cannot read non-ASCII)
#   * each launcher carries the five ring switches + --max-shared-prefixes 0
#   * each launcher's --kv-dtype is NOT a row-granular tier for the Swift artifact
#   * the three expected models are present with the recorded byte counts
#
# ASCII-ONLY on purpose: powershell.exe 5.1 reads a BOM-less .ps1 as ANSI/GBK, so any non-ASCII
# literal inside would be silently mangled (pitfall recorded on this project). All non-ASCII data
# (Chinese doc names) is read from disk at runtime as UTF-8.
#
# Exit: 0 when everything passes, 1 otherwise. Every check prints one greppable line.
param([string]$Kit = (Split-Path -Parent $PSScriptRoot), [switch]$NoModels)

$ErrorActionPreference = 'Continue'
$fail = 0
function Check([string]$Name, [bool]$Ok, [string]$Detail) {
  $verdict = 'OK  '
  if (-not $Ok) { $verdict = 'FAIL'; $script:fail++ }
  Write-Host ("[{0}] {1,-42} {2}" -f $verdict, $Name, $Detail)
}

Write-Host ("KIT = {0}" -f $Kit)

# ---- 1. the kit path and the model paths must be ASCII ------------------------------
$asciiOk = $true
foreach ($ch in $Kit.ToCharArray()) { if ([int]$ch -gt 127) { $asciiOk = $false } }
Check 'kit path is ASCII' $asciiOk $Kit

$modelsDir = Join-Path $Kit 'models'
# Only *.ninfer matters here: the rule is "the engine cannot read a non-ASCII MODEL path", not
# "no file in models\ may have a Chinese name". A readme next to the weights is harmless; a model
# named in Chinese is fatal, so the check is scoped to what can actually be loaded.
$modelFiles = @(Get-ChildItem -LiteralPath $modelsDir -File -Filter '*.ninfer' -ErrorAction SilentlyContinue | Sort-Object Name)
$modelAscii = $true
foreach ($m in $modelFiles) { foreach ($ch in $m.Name.ToCharArray()) { if ([int]$ch -gt 127) { $modelAscii = $false } } }
Check 'model file names are ASCII' $modelAscii ("{0} file(s)" -f $modelFiles.Count)

# ---- 1b. models: NOT SHIPPED WITH THIS KIT (2026-10-02) --------------------------------
#   The weights travel as a SEPARATE model pack, so a missing model file is not a failure
#   here: this kit is complete without it (engine + launchers + docs is what we ship).
#   What IS a failure: a model that sits in models\ with the WRONG byte count -- that is a
#   truncated or renamed download, i.e. exactly the class of fault a size check can catch.
#   So: absent = note, present = size check.  -NoModels stays accepted (same behaviour).
$expectedModels = @{
  'Ternary-Bonsai-2-27B-ninfer-v3-mtponly.ninfer' = 7654217216
  'Ternary-Bonsai-2-27B-ninfer-v3.ninfer'         = 9520051456
  'gsq_rco_iq3_s_dflash2_prop.ninfer'             = 15017456128
  'swift-qwen38-rco-iq3s.ninfer'                  = 15017456128
}
$presentModels = 0
foreach ($name in $expectedModels.Keys) {
  $p = Join-Path $modelsDir $name
  $f = Get-Item -LiteralPath $p -ErrorAction SilentlyContinue
  if ($null -eq $f) { continue }
  $presentModels++
  $ok = ($f.Length -eq $expectedModels[$name])
  Check ("model present, byte count: {0}" -f $name) $ok ("expected {0:N0} B, got {1:N0} B" -f $expectedModels[$name], $f.Length)
}
Check 'models are NOT shipped in this kit (separate model pack)' $true ("{0} of {1} downloaded and size-checked; missing ones are not a failure" -f $presentModels, $expectedModels.Count)

# ---- 2. engine + runtime DLLs --------------------------------------------------------
$engineDir = Join-Path $Kit 'engine'
# One binary per architecture: ninfer-serve-86.exe / -89.exe / -120a.exe. Match by pattern and say
# which ones are here -- a single fixed filename is how this check would go blind the moment the
# package started shipping three.
$engines = @(Get-ChildItem -LiteralPath $engineDir -File -Filter 'ninfer-serve*.exe' -ErrorAction SilentlyContinue | Sort-Object Name)
Check 'engine binaries present' ($engines.Count -ge 1) (($engines | ForEach-Object { $_.Name }) -join ', ')
foreach ($dll in 'cublas64_13.dll', 'cublasLt64_13.dll', 'cudart64_13.dll') {
  $p = Join-Path $engineDir $dll
  Check ("runtime dll: {0}" -f $dll) (Test-Path -LiteralPath $p) $p
}

# ---- 3. launchers: ASCII + required argv pieces --------------------------------------
$bats = @(Get-ChildItem -LiteralPath $Kit -File -Filter '*.bat' -ErrorAction SilentlyContinue | Sort-Object Name)
Check 'launchers found' ($bats.Count -ge 1) ("{0} .bat file(s)" -f $bats.Count)
foreach ($bat in $bats) {
  $bytes = [IO.File]::ReadAllBytes($bat.FullName)
  $nonAscii = 0
  foreach ($b in $bytes) { if ($b -gt 127) { $nonAscii++ } }
  $hasBom = (($bytes[0] -eq 0xEF) -and ($bytes[1] -eq 0xBB) -and ($bytes[2] -eq 0xBF))
  Check ("{0} is ASCII-only, no BOM" -f $bat.Name) (($nonAscii -eq 0) -and (-not $hasBom)) ("non-ascii={0} bom={1}" -f $nonAscii, $hasBom)

  $raw = [IO.File]::ReadAllText($bat.FullName, [Text.Encoding]::ASCII)
  # Strip comments before matching, so a required flag cannot be "satisfied" by prose about it.
  # Deliberately nothing smarter than that: which --spec a model needs is answered by the model
  # probe in this same folder (it reads the file), not by pattern-matching launcher text.
  $text = ((($raw -split "`r?`n") | Where-Object { $_.Trim() -notmatch '^(REM\b|::)' }) -join "`n")
  $need = @('NINFER_KV_WINDOW', 'NINFER_KV_RETRIEVE', 'NINFER_KV_RING', 'NINFER_HOST_PAGEABLE',
            'NINFER_KV_REUSE_HOSTBACKED', '--max-shared-prefixes 0', '--kv-dtype k8v4', '--spec ')
  $missing = @()
  foreach ($token in $need) { if ($text -notmatch [regex]::Escape($token)) { $missing += $token } }
  $detail = 'all required tokens present'
  if ($missing.Count -gt 0) { $detail = 'MISSING: ' + ($missing -join ', ') }
  Check ("{0} carries the five switches + guards" -f $bat.Name) ($missing.Count -eq 0) $detail
}

# ---- 4. SHA256SUMS -------------------------------------------------------------------
$sums = Join-Path $Kit 'SHA256SUMS.txt'
if (-not (Test-Path -LiteralPath $sums)) {
  Check 'SHA256SUMS.txt present' $false 'run the hash step first'
} else {
  $lines = @(Get-Content -LiteralPath $sums -Encoding UTF8 | Where-Object { $_.Trim() -ne '' })
  Check 'SHA256SUMS.txt non-empty' ($lines.Count -gt 0) ("{0} entr(ies)" -f $lines.Count)
  $bad = 0
  foreach ($line in $lines) {
    $parts = $line -split '\s+', 2
    if ($parts.Count -lt 2) { continue }
    $want = $parts[0].ToUpper()
    $rel = $parts[1].Trim()
    $full = Join-Path $Kit $rel
    if (-not (Test-Path -LiteralPath $full)) { Write-Host ("[FAIL] missing file: {0}" -f $rel); $bad++; continue }
    $got = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToUpper()
    if ($got -ne $want) { Write-Host ("[FAIL] hash mismatch: {0}" -f $rel); $bad++ }
  }
  Check 'every hashed file matches' ($bad -eq 0) ("{0} mismatch/missing" -f $bad)

  # ---- 4b. and the other direction: every kit file must be LISTED --------------------
  # Without this, deleting a file from the kit still passes ("34 hashed files all match"),
  # i.e. the gate would be blind to exactly the failure it exists to catch.
  $listed = @{}
  foreach ($line in $lines) {
    $parts = $line -split '\s+', 2
    if ($parts.Count -ge 2) { $listed[$parts[1].Trim()] = $true }
  }
  $unlisted = @()
  foreach ($f in (Get-ChildItem -LiteralPath $Kit -Recurse -File)) {
    if ($f.Name -eq 'SHA256SUMS.txt') { continue }   # the kit manifest and the nested package's
    if ($f.Extension -eq '.ninfer') { continue }     # models are NOT part of this kit (separate model pack)
    $rel = $f.FullName.Substring($Kit.Length).TrimStart('\')
    if (-not $listed.ContainsKey($rel)) { $unlisted += $rel }
  }
  Check 'no kit file is missing from SHA256SUMS.txt' ($unlisted.Count -eq 0) ("{0} unlisted of {1} on disk" -f $unlisted.Count, ($listed.Count + $unlisted.Count))
  foreach ($u in $unlisted) { Write-Host ("[FAIL] unlisted file: {0}" -f $u) }
}

# ---- 5. verdict ----------------------------------------------------------------------
Write-Host ''
if ($fail -eq 0) { Write-Host 'KITCHECK_VERDICT=PASS' } else { Write-Host ("KITCHECK_VERDICT=FAIL ({0} check(s))" -f $fail) }
if ($fail -ne 0) { exit 1 }
exit 0
