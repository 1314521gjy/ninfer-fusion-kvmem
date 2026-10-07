# Shared build-environment resolver for the verify scripts in this directory.
#
# WHY THIS FILE EXISTS
#   The first drafts of these scripts hard-coded one machine's paths (a VS install directory, a
#   scratch build directory, E:\cuda-13.3, a local vcpkg tree). A published criterion script that
#   carries the author's paths is not reproducible: the receiver cannot tell which of the values
#   matter, and a wrong one fails as "the flag did not take effect" rather than as an error.
#
#   Everything here is therefore DERIVED from the build directory being verified:
#     - the toolchain (cl.exe, ninja) from that build's CMakeCache.txt
#     - the CUDA root from CMAKE_CUDA_COMPILER
#     - the vcpkg root from CMAKE_PREFIX_PATH
#     - vcvars from vswhere, not from a guessed Visual Studio version directory
#   A value that cannot be derived is reported as unresolved instead of being defaulted silently.
#
# USAGE (dot-source it, then call the functions)
#   . "$PSScriptRoot\resolve-build-env.ps1"
#   $env = Resolve-BuildEnv -BuildDir $BuildDir
#   Use-MsvcEnv -BuildEnv $env
#
# Pure ASCII on purpose (Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI).

function Read-CacheValue {
    param([string]$CachePath, [string]$Key)
    if (-not (Test-Path $CachePath)) { return $null }
    $line = Select-String -Path $CachePath -Pattern ("^" + [regex]::Escape($Key) + ":[^=]*=") -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $line) { return $null }
    $text = $line.Line
    return $text.Substring($text.IndexOf('=') + 1).Trim()
}

function Find-VcVars {
    # Prefer vswhere: it reports the installed instance instead of guessing a year folder.
    $candidates = @()
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        $installs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        foreach ($install in $installs) {
            if ($install) { $candidates += (Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat') }
        }
    }
    foreach ($guess in @(
            "${env:ProgramFiles(x86)}\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
            "${env:ProgramFiles}\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
            "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")) {
        $candidates += $guess
    }
    foreach ($candidate in $candidates) { if ($candidate -and (Test-Path $candidate)) { return $candidate } }
    return $null
}

function Resolve-BuildEnv {
    param([Parameter(Mandatory = $true)][string]$BuildDir)
    $cache = Join-Path $BuildDir 'CMakeCache.txt'
    if (-not (Test-Path $cache)) {
        throw ("no CMakeCache.txt under " + $BuildDir + " -- point -BuildDir at a configured build tree")
    }
    $envInfo = [pscustomobject]@{
        BuildDir     = (Resolve-Path $BuildDir).Path
        CachePath    = $cache
        SourceDir    = (Read-CacheValue $cache 'CMAKE_HOME_DIRECTORY')
        Generator    = (Read-CacheValue $cache 'CMAKE_GENERATOR')
        NinjaExe     = (Read-CacheValue $cache 'CMAKE_MAKE_PROGRAM')
        CxxCompiler  = (Read-CacheValue $cache 'CMAKE_CXX_COMPILER')
        CudaCompiler = (Read-CacheValue $cache 'CMAKE_CUDA_COMPILER')
        PrefixPath   = (Read-CacheValue $cache 'CMAKE_PREFIX_PATH')
        VcVars       = (Find-VcVars)
    }
    # CUDA root = the parent of the bin directory holding nvcc.
    $envInfo | Add-Member -NotePropertyName CudaRoot -NotePropertyValue $null
    if ($envInfo.CudaCompiler) {
        $binDir = Split-Path $envInfo.CudaCompiler -Parent
        if ((Split-Path $binDir -Leaf) -eq 'bin') { $envInfo.CudaRoot = (Split-Path $binDir -Parent) }
    }
    $envInfo | Add-Member -NotePropertyName VcpkgRoot -NotePropertyValue $null
    if ($envInfo.PrefixPath -and ($envInfo.PrefixPath -match '^(.*?)[\\/]installed[\\/]')) {
        $envInfo.VcpkgRoot = $matches[1]
    }
    foreach ($field in @('SourceDir', 'NinjaExe', 'CxxCompiler', 'CudaCompiler', 'VcVars')) {
        if (-not $envInfo.$field) { Write-Host ("  [env] could not derive " + $field + " from " + $cache) }
    }
    return $envInfo
}

function Show-BuildEnv {
    param([Parameter(Mandatory = $true)]$BuildEnv)
    Write-Host ("  build dir   = " + $BuildEnv.BuildDir)
    Write-Host ("  source dir  = " + $BuildEnv.SourceDir)
    Write-Host ("  generator   = " + $BuildEnv.Generator)
    Write-Host ("  ninja       = " + $BuildEnv.NinjaExe)
    Write-Host ("  cl          = " + $BuildEnv.CxxCompiler)
    Write-Host ("  nvcc        = " + $BuildEnv.CudaCompiler)
    Write-Host ("  cuda root   = " + $BuildEnv.CudaRoot)
    Write-Host ("  vcpkg root  = " + $BuildEnv.VcpkgRoot)
    Write-Host ("  vcvars      = " + $BuildEnv.VcVars)
}

function Use-MsvcEnv {
    param([Parameter(Mandatory = $true)]$BuildEnv)
    if ($BuildEnv.VcVars) {
        $dump = & cmd /c ("call `"" + $BuildEnv.VcVars + "`" >nul 2>&1 && set")
        foreach ($line in $dump) {
            $eq = $line.IndexOf('=')
            if ($eq -gt 0) {
                $name = $line.Substring(0, $eq)
                $value = $line.Substring($eq + 1)
                if ($name -notmatch '[^A-Za-z0-9_()]') { Set-Item -Path ("env:" + $name) -Value $value -ErrorAction SilentlyContinue }
            }
        }
    }
    if ($BuildEnv.CudaRoot) { $env:CUDA_PATH = $BuildEnv.CudaRoot }
    if ($BuildEnv.VcpkgRoot) { $env:VCPKG_ROOT = $BuildEnv.VcpkgRoot }
    # The common vcpkg triplet name; a build verified with another one still resolves through
    # CMAKE_PREFIX_PATH, which the cache already recorded.
    $env:VCPKG_TARGET_TRIPLET = 'x64-windows'
    # A staged runtime needs the CUDA runtime DLLs on PATH before the first --help.
    if ($BuildEnv.CudaRoot) {
        $bins = @((Join-Path $BuildEnv.CudaRoot 'bin'), (Join-Path $BuildEnv.CudaRoot 'bin\x64')) |
                Where-Object { Test-Path $_ }
        if ($bins) { $env:PATH = (($bins -join ';') + ';' + $env:PATH) }
    }
}
