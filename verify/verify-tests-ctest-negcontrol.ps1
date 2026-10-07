# Test runner + negative control for the fused engine tree.
#
# WHY ctest AND NOT THE EXE PATHS (measured, not assumed):
#   Most tests in this project are NOT standalone executables. `ninfer_add_test()` compiles each
#   program as an OBJECT library with `main` renamed to `ninfer_bundle_entry_<name>` and links them
#   all into ONE binary, invoked as `ninfer_tests <program> [args...]`
#   (cmake/NinferBundles.cmake:8 and :39-41; tests\CMakeLists.txt:12). build.ninja shows the
#   per-test targets as phony over their .obj only (e.g. build.ninja:16609), so "run
#   ninfer_qwen3_5_context_store_test.exe" would find no such file. ctest is the only component that
#   knows the real command line and the staged-DLL working directory
#   (ninfer_stage_test_runtime_dlls(ninfer_tests), tests\CMakeLists.txt:17).
#
# CRITERIA (and why these, not "count the cases"):
#   The project has NO gtest -- measured: `gtest|TEST(` matches 0 times tree-wide. Tests are
#   hand-rolled and exit 0 (pass) / 1 (fail) / 77 (SKIP, see tests/guarded_main.h:42).
#   So the criteria are:
#     * tests run           > 0        (from ctest's own JUnit totals, not from stdout parsing)
#     * failures            == 0
#     * skipped             == 0       -> a skip is a machine-capacity skip, NEVER a pass
#   and the anti-false-green evidence is the NEGATIVE CONTROL (-NegativeControl): flip one row of the
#   classification table, rebuild that one program, and ctest MUST report it failed with the specific
#   assertion text. If it stays green, the test proves nothing and this script says so out loud.
#
# USAGE
#   pwsh -File run-tests.ps1
#   pwsh -File run-tests.ps1 -NegativeControl
#
# Pure ASCII on purpose (Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI).

param(
  # Point -BuildDir at a configured build tree; everything machine-specific (cl, ninja, CUDA root,
  # vcpkg root, vcvars) is derived from that build's CMakeCache.txt by resolve-build-env.ps1.
  [Parameter(Mandatory = $true)][string]$BuildDir,
  # Defaults to the source directory recorded in the build's own CMakeCache.
  [string]$SrcDir = '',
  # Defaults to a subdirectory beside this script.
  [string]$OutRoot = (Join-Path $PSScriptRoot 'out-tests'),
  # Anchored on purpose: a loose "vision_cpu" also matches ninfer_qwen3_5_vision_cpu_real_test,
  # which is a real-model test that returns 77 (skip) whenever no model artifact is configured
  # (tests/models/qwen3_5/test_vision_cpu_real.cpp). A skip is not a pass, so sweeping it in would
  # make the verdict permanently NOT-CLEAN and hide the four tests that actually matter here.
  [string]$Subset = '^(ninfer_failure_class_test|ninfer_qwen3_5_context_store_test|ninfer_qwen3_5_vision_cpu_test|ninfer_hadamard_transform_test)$',
  [switch]$NegativeControl
)

$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'resolve-build-env.ps1')
$buildEnv = Resolve-BuildEnv -BuildDir $BuildDir
$ninjaExe = $buildEnv.NinjaExe
if (-not $SrcDir) {
  $SrcDir = $buildEnv.SourceDir
  if (-not $SrcDir) { Write-Host "FATAL: -SrcDir not given and CMAKE_HOME_DIRECTORY is absent from the cache"; exit 2 }
}

function Invoke-CtestSubset([string]$regex, [string]$junitPath) {
  if (Test-Path $junitPath) { Remove-Item $junitPath -Force }
  $ctestOut = & ctest --test-dir $BuildDir -R $regex --output-on-failure --output-junit $junitPath -j 1 2>&1
  $ctestRc = $LASTEXITCODE
  $ctestOut | Set-Content -Path ($junitPath + '.console.txt') -Encoding utf8
  return [pscustomobject]@{ rc = $ctestRc; output = $ctestOut; junit = $junitPath }
}

function Read-CtestConsole([string[]]$lines) {
  # Fallback when --output-junit produced nothing this run: the console lines carry the same facts.
  #   "    1/5 Test   #2: ninfer_failure_class_test .............   Passed    0.01 sec"
  $cases = @()
  foreach ($line in $lines) {
    $m = [regex]::Match([string]$line, '^\s*\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.*\s*(\*{0,3})(Passed|Failed|Skipped|Timeout)\s+([\d.]+)\s+sec')
    if ($m.Success) {
      $status = $m.Groups[3].Value.ToLower()
      if ($m.Groups[2].Value.Length -gt 0 -and $status -eq 'failed') { $status = 'failed' }
      $cases += [pscustomobject]@{ name = $m.Groups[1].Value; seconds = $m.Groups[4].Value; status = $status; message = 'from ctest console' }
    }
  }
  return $cases
}

function Read-JUnit([string]$junitPath) {
  if (-not (Test-Path $junitPath)) { return $null }
  try { $xml = [xml](Get-Content $junitPath -Raw) } catch { return $null }
  $cases = @()
  foreach ($suite in $xml.testsuites.testsuite) {
    foreach ($case in $suite.testcase) {
      $status = 'passed'
      $message = ''
      if ($case.failure) { $status = 'failed'; $message = ([string]$case.failure.message) }
      elseif ($case.skipped) { $status = 'skipped'; $message = ([string]$case.skipped.message) }
      elseif ($case.'system-out' -match 'skip:') { $status = 'skipped(exit77)'; $message = 'skip marker in stdout' }
      $cases += [pscustomobject]@{ name = $case.name; seconds = $case.time; status = $status; message = ($message -replace '\s+', ' ') }
    }
  }
  return $cases
}

Show-BuildEnv -BuildEnv $buildEnv
Use-MsvcEnv -BuildEnv $buildEnv
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

Write-Host "=== test run  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host ("build dir = " + $BuildDir)
$bundle = Join-Path $BuildDir 'tests\ninfer_tests.exe'
Write-Host ("bundle exe present = " + (Test-Path $bundle))
if (Test-Path $bundle) {
  $sg = Join-Path (Split-Path $BuildDir -Parent) 'build-main-start.txt'
  if (Test-Path $sg) {
    $bs = [datetime]::Parse((Get-Content $sg -Raw).Trim())
    $fresh = (Get-Item $bundle).LastWriteTime -gt $bs
    Write-Host ("  freshness {0}  built {1} vs build start {2}" -f $(if ($fresh) { 'OK' } else { 'STALE' }), (Get-Item $bundle).LastWriteTime.ToString('HH:mm:ss'), $bs.ToString('HH:mm:ss'))
  }
}

Write-Host ""
Write-Host "--- registered tests (ctest -N) ---"
$listOut = & ctest --test-dir $BuildDir -N 2>&1
$listRc = $LASTEXITCODE
$total = 0
$m = [regex]::Match(($listOut -join "`n"), 'Total Tests:\s*(\d+)')
if ($m.Success) { $total = [int]$m.Groups[1].Value }
Write-Host ("  ctest -N exit = {0}   Total Tests = {1}" -f $listRc, $total)
Write-Host ("  subset regex = " + $Subset)

Write-Host ""
Write-Host "--- run subset ---"
$junit = Join-Path $OutRoot 'ctest-subset.xml'
$run = Invoke-CtestSubset $Subset $junit
Write-Host ("  ctest exit = " + $run.rc)
$cases = @(Read-JUnit $junit)
if ($cases.Count -eq 0) {
  Write-Host ("  NOTE: JUnit report unusable (file present = " + (Test-Path $junit) + "); parsing the ctest console instead")
  $cases = @(Read-CtestConsole $run.output)
}
if ($cases.Count -eq 0) {
  Write-Host "  FATAL: ctest produced no parseable results; raw console follows"
  $run.output | Select-Object -Last 40 | ForEach-Object { Write-Host ("    " + $_) }
  exit 4
}
Write-Host ("  cases reported = " + $cases.Count)
$cases | ForEach-Object { Write-Host ("    {0,-44} {1,-16} {2,7}s  {3}" -f $_.name, $_.status, $_.seconds, ($_.message.Substring(0, [math]::Min(90, $_.message.Length)))) }

$nPass = @($cases | Where-Object { $_.status -eq 'passed' }).Count
$nFail = @($cases | Where-Object { $_.status -eq 'failed' }).Count
$nSkip = @($cases | Where-Object { $_.status -like 'skipped*' }).Count
Write-Host ""
Write-Host ("--- summary: passed={0} failed={1} skipped={2} of {3} run ---" -f $nPass, $nFail, $nSkip, $cases.Count)
Write-Host "  NOTE: a skip (exit 77) is a machine-capacity skip, NOT a pass; it leaves behaviour unverified."

$subsetVerdict = if ($cases.Count -gt 0 -and $nFail -eq 0 -and $nSkip -eq 0) { 'PASS' } else { 'NOT-CLEAN' }
Write-Host ("VERDICT subset = " + $subsetVerdict)

$ncVerdict = 'not-run'
if ($NegativeControl) {
  Write-Host ""
  Write-Host "--- negative control: flip one row of the classification table ---"
  $header = Join-Path $SrcDir 'include\ninfer\failure_class.h'
  if (-not (Test-Path $header)) { Write-Host ("  FATAL: " + $header + " not found"); exit 5 }
  $backup = Join-Path $OutRoot 'failure_class.h.orig'
  Copy-Item $header $backup -Force
  $origHash = (Get-FileHash $backup -Algorithm SHA256).Hash
  Write-Host ("  original sha256 = " + $origHash)
  $text = [System.IO.File]::ReadAllText($header)
  $needle = 'if (dynamic_cast<const std::bad_alloc*>(&error) != nullptr) { return FailureClass::Capacity; }'
  $replace = 'if (dynamic_cast<const std::bad_alloc*>(&error) != nullptr) { return FailureClass::Other; }'
  if ($text.IndexOf($needle) -lt 0) {
    Write-Host "  FATAL: the row to flip is not present; nothing changed"
    exit 5
  }
  [System.IO.File]::WriteAllText($header, $text.Replace($needle, $replace))
  $flippedHash = (Get-FileHash $header -Algorithm SHA256).Hash
  Write-Host "  flipped: std::bad_alloc -> Other (was Capacity)"
  Write-Host ("  flipped sha256 = " + $flippedHash)
  Write-Host ("  differs from original = " + ($flippedHash -ne $origHash))
  if ($flippedHash -eq $origHash) {
    Write-Host "  FATAL: the flip never reached the file; nothing was tested"
    Copy-Item $backup $header -Force
    exit 6
  }

  # FORCE the rebuild. MEASURED 2026-10-07: after editing failure_class.h, ninja answered
  # "ninja: no work to do." -- this build uses CMake's scanned C++20 rules whose DEP_FILE is
  # <obj>.ddi.d (module dependencies), so a plain #included header is NOT an input of the object and
  # editing it triggers nothing. Trusting ninja here produced a green ctest on the OLD binary, i.e. a
  # negative control that tested nothing. Deleting the object and the exe makes the recompile explicit.
  $objDir = Join-Path $BuildDir 'tests\CMakeFiles\ninfer_failure_class_test.dir'
  Get-ChildItem $objDir -Filter 'test_failure_class.cpp.obj*' -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  Remove-Item (Join-Path $BuildDir 'tests\ninfer_failure_class_test.exe') -Force -ErrorAction SilentlyContinue
  Write-Host "  forced: removed the object + exe so the recompile cannot be skipped"

  $rb = & $ninjaExe -C $BuildDir 'tests\ninfer_failure_class_test.exe' 2>&1
  Write-Host ("  rebuild exit = " + $LASTEXITCODE)
  $rb | Select-Object -Last 3 | ForEach-Object { Write-Host ("    " + $_) }

  $jRed = Join-Path $OutRoot 'ctest-nc-flipped.xml'
  $redRun = Invoke-CtestSubset '^ninfer_failure_class_test$' $jRed
  $redCases = @(Read-JUnit $jRed)
  if ($redCases.Count -eq 0) { $redCases = @(Read-CtestConsole $redRun.output) }
  $redStatus = if ($redCases.Count -gt 0) { $redCases[0].status } else { 'unknown' }
  $redMsg = if ($redCases.Count -gt 0) { $redCases[0].message } else { '' }
  # The control is only a control if it fails for the RIGHT reason: the assertion that names the row
  # we flipped must be the one that fired. "It went red somehow" would also be satisfied by, say, a
  # compile error or a crash, which would prove nothing about this table.
  $expectText = 'std::bad_alloc must classify as Capacity'
  $sawExpect = (((($redRun.output) -join "`n")).IndexOf($expectText) -ge 0)
  Write-Host ("  expected assertion text present in output = " + $sawExpect)
  Write-Host ("  flipped-table ctest status = " + $redStatus)
  Write-Host ("  message: " + ($redMsg.Substring(0, [math]::Min(160, $redMsg.Length))))
  $wentRed = ($redStatus -eq 'failed') -and $sawExpect

  Write-Host "  restoring the original header (byte-exact) ..."
  Copy-Item $backup $header -Force
  $restoredHash = (Get-FileHash $header -Algorithm SHA256).Hash
  Write-Host ("  restored sha256 = " + $restoredHash + "  byte-exact=" + ($restoredHash -eq $origHash))
  if ($restoredHash -ne $origHash) { Write-Host "  FATAL: restore is not byte-exact; tree left dirty"; exit 6 }

  Get-ChildItem $objDir -Filter 'test_failure_class.cpp.obj*' -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  Remove-Item (Join-Path $BuildDir 'tests\ninfer_failure_class_test.exe') -Force -ErrorAction SilentlyContinue
  $rb2 = & $ninjaExe -C $BuildDir 'tests\ninfer_failure_class_test.exe' 2>&1
  Write-Host ("  rebuild-with-original exit = " + $LASTEXITCODE)
  Write-Host "  (same forced removal as above, so this rebuild is real too)"
  $jBack = Join-Path $OutRoot 'ctest-nc-restored.xml'
  $backRun = Invoke-CtestSubset '^ninfer_failure_class_test$' $jBack
  $backCases = @(Read-JUnit $jBack)
  if ($backCases.Count -eq 0) { $backCases = @(Read-CtestConsole $backRun.output) }
  $backStatus = if ($backCases.Count -gt 0) { $backCases[0].status } else { 'unknown' }
  Write-Host ("  restored-table ctest status = " + $backStatus)

  $ncVerdict = if ($wentRed -and $backStatus -eq 'passed') { 'PASS(goes red when flipped, comes back green when restored)' } else { 'FAIL(not a usable control)' }
  Write-Host ("--- negative control verdict: " + $ncVerdict + " ---")
}

$record = [pscustomobject]@{
  when = (Get-Date).ToString('o'); build_dir = $BuildDir; src_dir = $SrcDir
  ctest_total_registered = $total; subset_regex = $Subset
  subset_run = $cases.Count; passed = $nPass; failed = $nFail; skipped = $nSkip
  subset_verdict = $subsetVerdict; negative_control = $ncVerdict
  detail = ($cases | ForEach-Object { ($_.name + '=' + $_.status) }) -join '; '
}
$recordFile = Join-Path $OutRoot 'test-run.tsv'
$record | Export-Csv -Path $recordFile -Delimiter "`t" -NoTypeInformation -Encoding UTF8
Write-Host ("WROTE " + $recordFile)
Write-Host ("FINAL: subset=" + $subsetVerdict + "  negative_control=" + $ncVerdict)
