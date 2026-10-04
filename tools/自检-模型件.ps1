# model-check.ps1 -- inspect ONE .ninfer model file BEFORE blaming the engine.
#
# WHY THIS EXISTS: the failure that costs the most time is not a crash, it is a model file that
# loads fine and runs SLOW, because it is a "bare" repack with no speculative head. Measured on
# this project: the same prompt on the same card ran 71.9 tok/s bare vs 276.8 tok/s with the
# dflash2 head (3.9x). Adding --spec to a bare file does not degrade -- it refuses to start
# ("Qwen3.5 config: missing component mtp"). So the question "which --spec may I pass?" must be
# answered from the FILE, not from trial and error.
#
# Second job: catch a truncated download. The header states the manifest length and the payload
# length, so "file is 3 GB short" is a decision, not a guess.
#
# The container layout read here (v3):
#   [0..6]   magic "NINFER\0"     [7] version byte    [8..15] manifest length, uint64 LE
#   [16..31] artifact id, 16 bytes                    [32..] manifest JSON, then the payload
#   payload starts at align_up(32 + manifest_len, 4096)
#
# ASCII-only on purpose (powershell.exe 5.1 reads a BOM-less .ps1 as ANSI, so a non-ASCII byte
# here would be mangled). All output lines are KEY=VALUE on purpose: greppable by an agent.
#
# Usage: powershell -ExecutionPolicy Bypass -File model-check.ps1 -Model <file.ninfer>
#        [-ExpectSha <hex>] [-ExpectBytes <int>] [-Quiet]
# Exit: 0 = ok   3 = readable but NO speculative head (bare; expect ~3.9x slower)
#       1 = broken/truncated   2 = could not read the file at all
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [string]$ExpectSha = '',
  [long]$ExpectBytes = 0,
  [int]$VramGb = 0,
  [switch]$Quiet
)

$ErrorActionPreference = 'Continue'
$fail = 0
function Emit([string]$k, [string]$v) { Write-Host ($k + '=' + $v) }

if (-not (Test-Path -LiteralPath $Model)) {
  Emit 'MODEL_FILE' $Model
  Emit 'MODELCHECK_VERDICT' 'FAIL'
  Emit 'MODELCHECK_REASON' 'file not found'
  exit 2
}
$item = Get-Item -LiteralPath $Model
Emit 'MODEL_FILE' $item.FullName
Emit 'MODEL_BYTES' ($item.Length.ToString())

# --- ASCII path gate: the native exe cannot read a non-ASCII path at all -----------------
$na = 0
foreach ($ch in $item.FullName.ToCharArray()) { if ([int]$ch -gt 127) { $na++ } }
Emit 'PATH_NON_ASCII_CHARS' ($na.ToString())
if ($na -gt 0) { Emit 'PATH_WARNING' 'engine will fail with invalid UTF-8 byte; move the file to an ASCII path' }

# --- size + hash ---------------------------------------------------------------------------
if ($ExpectBytes -gt 0) {
  $sizeOk = ($item.Length -eq $ExpectBytes)
  Emit 'SIZE_EXPECTED' ($ExpectBytes.ToString())
  Emit 'SIZE_MATCH' ($(if ($sizeOk) { 'YES' } else { 'NO' }))
  if (-not $sizeOk) { $fail = 1 }
}
if ($ExpectSha -ne '') {
  $got = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash
  Emit 'SHA256' $got
  $shaOk = ($got -eq $ExpectSha.ToUpper())
  Emit 'SHA256_EXPECTED' $ExpectSha.ToUpper()
  Emit 'SHA256_MATCH' ($(if ($shaOk) { 'YES' } else { 'NO' }))
  if (-not $shaOk) { $fail = 1 }
} else {
  Emit 'SHA256' (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash
}

# --- container header ----------------------------------------------------------------------
$fs = $null
try { $fs = New-Object IO.FileStream($item.FullName, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite) }
catch {
  Emit 'MODELCHECK_VERDICT' 'FAIL'
  Emit 'MODELCHECK_REASON' ('cannot open: ' + $_.Exception.Message)
  exit 2
}
try {
  $head = New-Object byte[] 32
  $read = $fs.Read($head, 0, 32)
  if ($read -lt 32) {
    Emit 'HEADER_BYTES_READ' ($read.ToString())
    Emit 'MODELCHECK_VERDICT' 'FAIL'
    Emit 'MODELCHECK_REASON' 'file is smaller than the 32-byte header (truncated download?)'
    $fs.Close(); exit 1
  }
  $magic = [Text.Encoding]::ASCII.GetString($head, 0, 7)
  $version = $head[7]
  $manifestLen = [BitConverter]::ToUInt64($head, 8)
  $artifactId = ([BitConverter]::ToString($head, 16, 16)).Replace('-', '').ToLower()
  Emit 'CONTAINER_MAGIC' ($magic -replace "`0", '')
  Emit 'CONTAINER_VERSION' ($version.ToString())
  Emit 'MANIFEST_BYTES' ($manifestLen.ToString())
  Emit 'ARTIFACT_ID' $artifactId

  if ($magic -ne "NINFER`0") {
    Emit 'MODELCHECK_VERDICT' 'FAIL'
    Emit 'MODELCHECK_REASON' 'not a NInfer container (magic mismatch) -- wrong file, or a web page saved with a .ninfer name'
    $fs.Close(); exit 1
  }
  if ($version -ne 3) {
    Emit 'MODELCHECK_VERDICT' 'FAIL'
    Emit 'MODELCHECK_REASON' ('container version ' + $version + ' -- this engine accepts v3 only; a v2 file must be converted first')
    $fs.Close(); exit 1
  }

  $mbuf = New-Object byte[] ([int]$manifestLen)
  $got = 0
  while ($got -lt [int]$manifestLen) {
    $n = $fs.Read($mbuf, $got, ([int]$manifestLen - $got))
    if ($n -le 0) { break }
    $got += $n
  }
  if ($got -lt [int]$manifestLen) {
    Emit 'MANIFEST_BYTES_READ' ($got.ToString())
    Emit 'MODELCHECK_VERDICT' 'FAIL'
    Emit 'MODELCHECK_REASON' 'manifest is cut short -- truncated download'
    $fs.Close(); exit 1
  }

  $payloadOffset = [int]([Math]::Ceiling((32 + [double]$manifestLen) / 4096.0) * 4096)
  Emit 'PAYLOAD_OFFSET' ($payloadOffset.ToString())

  $manifest = $null
  try { $manifest = [Text.Encoding]::UTF8.GetString($mbuf) | ConvertFrom-Json }
  catch {
    Emit 'MODELCHECK_VERDICT' 'FAIL'
    Emit 'MODELCHECK_REASON' ('manifest is not valid JSON: ' + $_.Exception.Message)
    $fs.Close(); exit 1
  }

  # payload length declared in the manifest vs what is on disk => the truncation test.
  # It lives in files[0].payload_bytes (the directory's required members are
  # components/objects/bindings/uses/files); a top-level payload_bytes is accepted too, because
  # guessing one shape and reporting "ABSENT" would silently skip the truncation test entirely.
  $declared = $null
  $declaredWhere = 'ABSENT'
  if ($manifest.PSObject.Properties.Name -contains 'payload_bytes') {
    $declared = [int64]$manifest.payload_bytes; $declaredWhere = 'top-level'
  } elseif (($manifest.PSObject.Properties.Name -contains 'files') -and (@($manifest.files).Count -gt 0)) {
    $f0 = @($manifest.files)[0]
    if ($f0.PSObject.Properties.Name -contains 'payload_bytes') {
      $declared = [int64]$f0.payload_bytes; $declaredWhere = 'files[0]'
    }
  }
  Emit 'PAYLOAD_BYTES_SOURCE' $declaredWhere
  if ($declared -ne $null) {
    $expectedTotal = [int64]$payloadOffset + $declared
    Emit 'PAYLOAD_BYTES_DECLARED' ($declared.ToString())
    Emit 'FILE_BYTES_EXPECTED' ($expectedTotal.ToString())
    $lenOk = ($item.Length -eq $expectedTotal)
    Emit 'LENGTH_MATCH' ($(if ($lenOk) { 'YES' } else { 'NO' }))
    if (-not $lenOk) {
      Emit 'LENGTH_DELTA_BYTES' (($item.Length - $expectedTotal).ToString())
      $fail = 1
    }
  } else {
    Emit 'PAYLOAD_BYTES_DECLARED' 'ABSENT'
    if ($item.Length -lt $payloadOffset) {
      Emit 'MODELCHECK_VERDICT' 'FAIL'
      Emit 'MODELCHECK_REASON' 'file is smaller than header+manifest -- truncated'
      $fs.Close(); exit 1
    }
  }

  # --- components: this is the answer to "which --spec may I pass?" ----------------------
  $names = @()
  if ($manifest.PSObject.Properties.Name -contains 'components') {
    $names = @($manifest.components.PSObject.Properties.Name)
  }
  Emit 'COMPONENTS' (($names | Sort-Object) -join ',')
  foreach ($n in ($names | Sort-Object)) {
    $sub = @($manifest.components.$n.PSObject.Properties.Name | Sort-Object)
    Emit ('COMPONENT_' + $n.ToUpperInvariant() + '_KEYS') ($sub -join ',')
  }
  $hasMtp    = ($names -contains 'mtp')
  $hasDflash = ($names -contains 'dflash2')
  $hasDflash1 = ($names -contains 'dflash')
  $hasVision = ($names -contains 'vision')
  Emit 'HAS_MTP'     ($(if ($hasMtp) { 'YES' } else { 'NO' }))
  Emit 'HAS_DFLASH2' ($(if ($hasDflash) { 'YES' } else { 'NO' }))
  Emit 'HAS_VISION'  ($(if ($hasVision) { 'YES' } else { 'NO' }))

  # the proposal head lives as a sub-key of the text component, not as a component of its own
  $hasProposal = $false
  if ($names -contains 'text') {
    $hasProposal = (@($manifest.components.text.PSObject.Properties.Name) -contains 'proposal')
  }
  Emit 'HAS_PROPOSAL_HEAD' ($(if ($hasProposal) { 'YES' } else { 'NO' }))

  $choices = @()
  if ($hasDflash) { $choices += 'dflash2' }
  if ($hasMtp)    { $choices += 'mtp' }
  if ($choices.Count -eq 0) { $choices += 'none' }
  Emit 'SPEC_CHOICES' ($choices -join ',')

  # ---- small-card rule: at 10 GB and under, MTP is the ONLY head that fits ---------------
  # The dflash2 head is 1.3 GiB (ternary tier) to 2.07 GiB (GSQ/Swift tiers) of extra resident
  # weights plus its own draft workspace; the MTP head is 332-430 MiB. On a small card that
  # difference is the difference between starting and not starting, so the recommendation is not
  # a menu here -- it is one answer.
  $smallCard = ($VramGb -gt 0 -and $VramGb -le 10)
  Emit 'VRAM_GB' ($(if ($VramGb -gt 0) { $VramGb.ToString() } else { 'UNKNOWN' }))
  if ($smallCard) {
    Emit 'SMALL_CARD_RULE' 'USE --spec mtp ONLY. Do NOT enable dflash2.'
    Emit 'SMALL_CARD_WHY' 'dflash2 adds 1.3-2.07 GiB of resident draft weights + a draft workspace; the mtp head costs 332-430 MiB'
    Emit 'SMALL_CARD_MUST_NOT' '--spec dflash2 / --lm-head-draft-on-dflash2 / --vision / a large --kv-capacity'
  }

  if ($hasDflash -and $hasProposal -and -not $smallCard) {
    Emit 'RECOMMENDED_ARGV' '--spec dflash2 --draft-tokens 4 --lm-head-draft'
    Emit 'ACCELERATION' 'DFLASH2'
  } elseif ($hasMtp -and $hasProposal) {
    # mtp + proposal head is the small-card answer, and it is also the fallback when there is no
    # dflash2 at all. --lm-head-draft is legal here because the proposal head is present.
    Emit 'RECOMMENDED_ARGV' '--spec mtp --draft-tokens 4 --lm-head-draft'
    Emit 'ACCELERATION' 'MTP_ONLY'
    if ($smallCard -and $hasDflash) {
      Emit 'SMALL_CARD_NOTE' 'this file also has dflash2; on your card do not use it'
    }
  } elseif ($hasDflash) {
    Emit 'RECOMMENDED_ARGV' '--spec dflash2 --draft-tokens 4   (no proposal head: --lm-head-draft will refuse to start)'
    Emit 'ACCELERATION' 'DFLASH2_NO_PROPOSAL'
    if ($smallCard) {
      Emit 'SMALL_CARD_WARNING' 'this file has NO mtp component and your card is small: dflash2 is the only head it carries and it may not fit'
    }
  } elseif ($hasMtp) {
    Emit 'RECOMMENDED_ARGV' '--spec mtp --draft-tokens 4   (no proposal head: --lm-head-draft will refuse to start)'
    Emit 'ACCELERATION' 'MTP_NO_PROPOSAL'
  } else {
    Emit 'RECOMMENDED_ARGV' '(no --spec at all)'
    Emit 'ACCELERATION' 'NONE'
    Emit 'BARE_WARNING' 'this file carries no speculative head: do NOT pass --spec (it refuses to start), and expect roughly 3.9x slower decode than the same model with the dflash2 head'
  }

  # object / binding / use counts -- a quick sanity read on the graph the loader walks.
  # NOTE: `bindings` is a JSON OBJECT (name -> {object/parts}), not an array, so @(...).Count on it
  # returns 1 and reads as a plausible-looking number. Count its properties instead.
  foreach ($k in 'objects', 'bindings', 'uses') {
    $c = 'ABSENT'
    if ($manifest.PSObject.Properties.Name -contains $k) {
      $v = $manifest.$k
      if ($v -is [System.Array]) { $c = $v.Count.ToString() }
      elseif ($v -is [System.Management.Automation.PSCustomObject]) {
        $c = (@($v.PSObject.Properties.Name)).Count.ToString() + ' (object)'
      } else { $c = '1' }
    }
    Emit ('MANIFEST_' + $k.ToUpperInvariant()) $c
  }
} finally {
  if ($fs) { $fs.Close() }
}

if ($fail -ne 0) {
  Emit 'MODELCHECK_VERDICT' 'FAIL'
  Emit 'MODELCHECK_REASON' 'size/hash/length mismatch against the values you passed in'
  exit 1
}
if (-not $hasMtp -and -not $hasDflash -and -not $hasDflash1) {
  Emit 'MODELCHECK_VERDICT' 'BARE'
  Emit 'MODELCHECK_REASON' 'readable v3 container, but no speculative component -- usable, just slow'
  exit 3
}
Emit 'MODELCHECK_VERDICT' 'PASS'
exit 0
