# CGTKEY - Windows build helper.
#
# The Makefile is the reference build, but Windows boxes rarely have make and
# nvcc needs the MSVC environment loaded before it will run. This script finds
# Visual Studio, loads it, and runs the same compile the Makefile does.
#
#   .\build.ps1                 fat binary for every arch this nvcc supports
#   .\build.ps1 -Arch 89        single arch (fast rebuild while developing)
#   .\build.ps1 -Tune "-DCGT_ROUNDS=16"
#   .\build.ps1 -SelfTest       build and run the host self-test too

param(
  [string]$Arch     = "",
  [string]$Tune     = "",
  [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# --- locate nvcc -----------------------------------------------------------
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
  Write-Host "[!] nvcc not on PATH. Install the CUDA Toolkit." -ForegroundColor Red
  exit 1
}

# --- locate a usable MSVC environment ---------------------------------------
# nvcc shells out to cl.exe on Windows and will not find it unless vcvars64 has
# run. Complication: nvcc hard-rejects host compilers newer than its toolkit
# knows about, so "newest Visual Studio" is often the wrong answer - a box with
# VS 2022 and a newer VS side by side must use the 2022 one. Rather than
# hardcoding a version cap that goes stale with every CUDA release, collect all
# the installs and probe each with a throwaway compile, newest first. The first
# one nvcc accepts is the one we use.
function Get-VSCandidates {
  $out = @()
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhere) {
    $found = & $vswhere -legacy -prerelease -products * `
               -format value -property installationPath 2>$null
    foreach ($p in $found) {
      if ($p) {
        $cand = Join-Path $p "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $cand) { $out += $cand }
      }
    }
  }
  if (-not $out) {
    foreach ($year in @("2022", "2019", "2017")) {
      foreach ($ed in @("Enterprise", "Professional", "Community", "BuildTools")) {
        foreach ($pf in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
          $cand = "$pf\Microsoft Visual Studio\$year\$ed\VC\Auxiliary\Build\vcvars64.bat"
          if (Test-Path $cand) { $out += $cand }
        }
      }
    }
  }
  # Newest first, so a supported-and-recent toolchain wins over an ancient one.
  return ($out | Sort-Object -Descending -Unique)
}

function Test-VCVars($vcvars) {
  # Trivial device compile: fails fast on an unsupported host compiler without
  # paying for the real kernel. Done in a scratch dir under the script root so
  # the probe never has to reason about oddly-formed TEMP paths.
  $dir = Join-Path $PSScriptRoot ".cgtprobe"
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $probe = Join-Path $dir "probe.cu"
  $obj   = Join-Path $dir "probe.obj"
  Set-Content -LiteralPath $probe -Encoding ascii -Value @'
__global__ void k(int *p) { *p = 1; }
'@
  cmd /c "`"$vcvars`" >nul 2>&1 && nvcc -c `"$probe`" -o `"$obj`" >nul 2>&1"
  $good = ($LASTEXITCODE -eq 0)
  Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
  return $good
}

$candidates = Get-VSCandidates
if (-not $candidates) {
  Write-Host "[!] Visual Studio C++ tools not found. nvcc needs cl.exe." -ForegroundColor Red
  Write-Host "    Install 'Desktop development with C++' from the VS installer."
  exit 1
}

$vcvars = $null
foreach ($c in $candidates) {
  if (Test-VCVars $c) { $vcvars = $c; break }
  Write-Host "[*] Skipping host compiler nvcc rejects: $c" -ForegroundColor DarkGray
}
if (-not $vcvars) {
  Write-Host "[!] No installed Visual Studio is compatible with this CUDA Toolkit." -ForegroundColor Red
  Write-Host "    Install VS 2022 (or the version your CUDA release supports)."
  exit 1
}
Write-Host "[*] Host compiler: $vcvars"

# --- architectures ---------------------------------------------------------
if ($Arch) {
  $gencode = "-gencode=arch=compute_$Arch,code=sm_$Arch"
  Write-Host "[*] Building for sm_$Arch only."
} else {
  # Ask nvcc what it supports rather than hardcoding: the answer differs by
  # toolkit version, and a stale list fails to build on newer or older ones.
  $archs = (& nvcc --list-gpu-arch) -replace 'compute_', '' |
             Where-Object { $_ -match '^\d+$' } |
             ForEach-Object { [int]$_ } | Sort-Object
  if (-not $archs) { $archs = @(50, 52, 60, 61, 70, 75, 80, 86) }
  $newest = $archs[-1]
  $gencode = (($archs | ForEach-Object {
                "-gencode=arch=compute_$_,code=sm_$_" }) -join ' ') +
             " -gencode=arch=compute_$newest,code=compute_$newest"
  Write-Host "[*] Building fat binary for: $($archs -join ', ') (+PTX for $newest)."
  Write-Host "[*] This compiles the kernel once per arch and takes a few minutes."
}

$cxxSrc  = "cgtcli.cpp cgtmath.cpp cgtdigest.cpp cgtpool.cpp cgtpkpool.cpp cgtspan.cpp"
$cuSrc   = "cgtgpu.cu"
$flags   = "-O3 -std=c++14 -Xptxas -O3 --use_fast_math --extra-device-vectorization"

$cmd = "`"$vcvars`" >nul 2>&1 && nvcc $flags $gencode $Tune $cxxSrc $cuSrc -o cgtkey.exe"
Write-Host "[*] Compiling..."
$sw = [Diagnostics.Stopwatch]::StartNew()
cmd /c $cmd
if ($LASTEXITCODE -ne 0) {
  Write-Host "[!] Build failed." -ForegroundColor Red
  exit $LASTEXITCODE
}
$sw.Stop()
Write-Host ("[+] Built cgtkey.exe in {0:N0}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green

if ($SelfTest) {
  Write-Host "[*] Building host self-test..."
  $testSrc = "cgtcheck.cpp cgtmath.cpp cgtdigest.cpp cgtpool.cpp cgtpkpool.cpp cgtspan.cpp"
  cmd /c "`"$vcvars`" >nul 2>&1 && cl /nologo /O2 /EHsc /std:c++14 $Tune $testSrc /Fe:cgtcheck.exe >nul"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Self-test build failed." -ForegroundColor Red
    exit $LASTEXITCODE
  }
  & .\cgtcheck.exe
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
