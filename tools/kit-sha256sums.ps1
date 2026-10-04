# kit-sha256sums.ps1 -- regenerate <kit>\SHA256SUMS.txt, WITHOUT re-hashing 40 GB every time.
#
# The bug this fixes: the first version hashed the whole kit on every run. The three model files are
# 38.8 of the 38.84 GiB, so editing one 8 KB doc cost a full 40 GB scan. That is not diligence, it
# is waste -- and it is the kind of "verification" that burns a project's time without catching
# anything a cheaper check would not catch.
#
# How it decides what to hash:
#   1. cache hit   -- <sidecar cache> says this path has this exact size AND mtime  => reuse hash
#   2. seed hit    -- the previous SHA256SUMS.txt already has this path, and the file has not been
#                     written since that manifest was generated                  => reuse hash
#   3. otherwise   -- hash it (this is the only case that touches 15 GB)
# -Full forces case 3 for everything (use it when you actually suspect tampering, not "just in case").
#
# The kit's own gate (engine\<kit self-check>.ps1) still re-hashes every listed file from scratch --
# that one runs ONCE, on the receiving machine, and is the check that matters to them.
#
# Kept OUTSIDE the kit: a generator inside it would change the manifest it writes.
# ASCII-only (powershell.exe 5.1 reads a BOM-less .ps1 as ANSI; the paths here are Chinese).
param(
  [string]$Kit = 'E:\betakit-20260930',
  [string]$Cache = 'E:\infer-build\.kit-sums-cache.tsv',
  [switch]$Full
)

$ErrorActionPreference = 'Continue'
if (-not (Test-Path -LiteralPath $Kit)) { Write-Host "REFUSE: kit not found: $Kit"; exit 3 }

$sumsPath = Join-Path $Kit 'SHA256SUMS.txt'
$sw = [Diagnostics.Stopwatch]::StartNew()

# ---- previous manifest: path -> hash ------------------------------------------------------
$prev = @{}
$prevTime = [datetime]::MinValue
if (Test-Path -LiteralPath $sumsPath) {
  $prevTime = (Get-Item -LiteralPath $sumsPath).LastWriteTimeUtc
  foreach ($line in [IO.File]::ReadAllLines($sumsPath)) {
    if ($line.Trim() -eq '') { continue }
    $p = $line -split '\s+', 2
    if ($p.Count -ge 2) { $prev[$p[1].Trim()] = $p[0].ToUpper() }
  }
}

# ---- sidecar cache: path -> size|mtimeTicks|hash ------------------------------------------
$cacheMap = @{}
if ((Test-Path -LiteralPath $Cache) -and (-not $Full)) {
  foreach ($line in [IO.File]::ReadAllLines($Cache)) {
    if ($line.Trim() -eq '') { continue }
    $c = $line -split "`t"
    if ($c.Count -ge 4) { $cacheMap[$c[0]] = @{ Size = $c[1]; Ticks = $c[2]; Hash = $c[3] } }
  }
}

$files = @(Get-ChildItem -LiteralPath $Kit -Recurse -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' })
Write-Host ("kit   = {0}" -f $Kit)
Write-Host ("files = {0}   previous manifest entries = {1}   cache entries = {2}" -f $files.Count, $prev.Count, $cacheMap.Count)

$rows = @()
$newCache = @()
$reused = 0; $fromManifest = 0; $hashed = 0; $hashedBytes = 0L
foreach ($f in $files) {
  $rel = $f.FullName.Substring($Kit.Length).TrimStart('\')
  $size = $f.Length.ToString()
  $ticks = $f.LastWriteTimeUtc.Ticks.ToString()
  $hash = $null

  if ($cacheMap.ContainsKey($rel)) {
    $c = $cacheMap[$rel]
    if ($c.Size -eq $size -and $c.Ticks -eq $ticks -and $c.Hash -match '^[0-9A-F]{64}$') {
      $hash = $c.Hash; $reused++
    }
  }
  if (-not $hash -and (-not $Full) -and $prev.ContainsKey($rel) -and $f.LastWriteTimeUtc -lt $prevTime) {
    # untouched since the manifest that already recorded its hash was written
    $hash = $prev[$rel]; $fromManifest++
  }
  if (-not $hash) {
    $hash = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash
    $hashed++; $hashedBytes += $f.Length
  }
  $rows += [pscustomobject]@{ Rel = $rel; Hash = $hash }
  $newCache += ($rel + "`t" + $size + "`t" + $ticks + "`t" + $hash)
}

$lines = @($rows | Sort-Object Rel | ForEach-Object { "{0}  {1}" -f $_.Hash, $_.Rel })
$enc = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllLines($sumsPath, $lines, $enc)
[IO.File]::WriteAllLines($Cache, $newCache, $enc)

$sw.Stop()
Write-Host ("hash sources: cache {0} + manifest-seed {1} + freshly hashed {2} ({3:N2} GiB)" -f `
    $reused, $fromManifest, $hashed, ($hashedBytes / 1GB))
Write-Host ("wrote {0}  ({1} line(s))   {2:N1} s" -f $sumsPath, $lines.Count, $sw.Elapsed.TotalSeconds)
Write-Host ("cache {0}" -f $Cache)
$b = [IO.File]::ReadAllBytes($sumsPath)
Write-Host ("BOM = {0}" -f (($b[0] -eq 0xEF) -and ($b[1] -eq 0xBB) -and ($b[2] -eq 0xBF)))

# ---- cheap alignment check (a set comparison; proves the "unlisted file" failure is gone) ---
$listed = @{}
foreach ($l in $lines) { $p = $l -split '\s+', 2; if ($p.Count -ge 2) { $listed[$p[1].Trim()] = $true } }
$onDisk = @($files | ForEach-Object { $_.FullName.Substring($Kit.Length).TrimStart('\') })
$unlisted = @($onDisk | Where-Object { -not $listed.ContainsKey($_) })
$ghost = @($listed.Keys | Where-Object { $onDisk -notcontains $_ })
Write-Host ("ALIGN listed={0} ondisk={1} unlisted={2} ghost={3}" -f $listed.Count, $onDisk.Count, $unlisted.Count, $ghost.Count)
foreach ($u in $unlisted) { Write-Host ("  UNLISTED {0}" -f $u) }
foreach ($g in $ghost) { Write-Host ("  GHOST    {0}" -f $g) }
exit $(if ($unlisted.Count -eq 0 -and $ghost.Count -eq 0) { 0 } else { 1 })
