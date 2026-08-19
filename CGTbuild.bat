@echo off
REM ============================================================================
REM  CGTbuild.bat - self-configuring, runtime-verified build for cgtkey.
REM
REM  Just run it. No paths to edit. It auto-detects:
REM    * CUDA        (via CUDA_PATH, else newest install under Program Files)
REM    * Visual Studio + the KNOWN-GOOD MSVC toolset 14.29.30133 that CUDA 13.1
REM      compiles cleanly with (picks whichever VS install actually has it, so a
REM      newer VS 18/2026 that lacks it is skipped rather than causing crashes).
REM  The device compiler (cicc) crashes non-deterministically, so it retries the
REM  device compile until a RUNTIME smoke test (finds a known oracle key) passes
REM  - a linkable .obj can still be broken, so linking alone is not enough.
REM
REM    %1 = CGT_STRIDE_HALF     (default 2048)
REM    %2 = max compile retries (default 12)
REM    %3 = GPU arch            (default sm_120)  e.g. sm_89, sm_86
REM  On success: builds cgtkey.exe AND copies it to cgtkey_<STRIDE_HALF>.exe.
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM ---------------- parameters ----------------
set "SH=%~1"
set "MAXR=%~2"
set "ARCH=%~3"
if "%SH%"==""   set "SH=2048"
if "%MAXR%"=="" set "MAXR=12"
if "%ARCH%"=="" set "ARCH=sm_120"
for /f "tokens=2 delims=_" %%a in ("%ARCH%") do set "CC=%%a"
if "%CC%"=="" set "CC=120"

set "PREF_TOOLSET=14.29.30133"
set "PK=0306f30628b27e66f3f5eec43b6a4c7385b76e919250a5dd768f0463592bd5b658"
set "RANGE=5ABCD0000:5ABCE0000"

echo(
echo [*] CGTbuild: STRIDE_HALF=%SH%  arch=%ARCH%  retries=%MAXR%

REM ================= locate CUDA =================
set "CUDADIR="
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "CUDADIR=%CUDA_PATH%"
if not defined CUDADIR if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" set "CUDADIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
if not defined CUDADIR (
  for /f "delims=" %%d in ('dir /b /ad /o-n "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do (
    if not defined CUDADIR if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%d\bin\nvcc.exe" set "CUDADIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%d"
  )
)
if not defined CUDADIR (
  echo [!] CUDA toolkit not found.
  echo     Install CUDA ^(v13.1 recommended^) or set the CUDA_PATH environment variable.
  exit /b 1
)
echo [*] CUDA    : %CUDADIR%
set "PATH=%CUDADIR%\bin;%CUDADIR%\nvvm\bin;%PATH%"
set "CUDA_PATH=%CUDADIR%"

REM ================= locate Visual Studio + known-good toolset =================
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDIR="
set "TOOLSET="
if exist "%VSWHERE%" (
  REM Prefer an install that actually HAS the known-good toolset.
  for /f "usebackq delims=" %%i in (`"%VSWHERE%" -products * -property installationPath 2^>nul`) do (
    if not defined VSDIR if exist "%%i\VC\Tools\MSVC\%PREF_TOOLSET%" (
      set "VSDIR=%%i"
      set "TOOLSET=%PREF_TOOLSET%"
    )
  )
  REM None had it: fall back to the latest install with the default toolset.
  if not defined VSDIR for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -property installationPath 2^>nul`) do set "VSDIR=%%i"
)
REM Last-ditch fixed-path fallback if vswhere is absent.
if not defined VSDIR (
  for %%p in (
    "C:\Program Files\Microsoft Visual Studio\2022\Community"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools"
  ) do if not defined VSDIR if exist "%%~p\VC\Tools\MSVC\%PREF_TOOLSET%" (set "VSDIR=%%~p" & set "TOOLSET=%PREF_TOOLSET%")
)
if not defined VSDIR (
  echo [!] Visual Studio with C++ tools not found.
  echo     Install "Visual Studio 2022" with the "Desktop development with C++" workload.
  exit /b 1
)
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" (
  echo [!] Found VS at "%VSDIR%" but vcvars64.bat is missing.
  echo     Add the "Desktop development with C++" workload in the VS Installer.
  exit /b 1
)
echo [*] VS      : %VSDIR%

if defined TOOLSET (
  echo [*] Toolset : %TOOLSET%  ^(known-good for CUDA 13.1^)
  call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=%TOOLSET% >nul 2>nul
) else (
  echo [*] Toolset : default  ^(known-good %PREF_TOOLSET% not installed; relying on -allow-unsupported-compiler^)
  call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
)
where cl >nul 2>nul
if errorlevel 1 (
  echo [!] MSVC compiler 'cl' is not on PATH after environment setup.
  echo     The VS C++ toolset may be incomplete - repair it in the VS Installer.
  exit /b 1
)

REM ================= build =================
if exist cgtkey.exe del /f cgtkey.exe
del /q *.obj 2>nul

echo [*] Stage 1: host .cpp at /Od ...
cl /nologo /c /Od /EHsc /std:c++14 /DCGT_STRIDE_HALF=%SH% /I "%CUDADIR%\include" ^
   cgtcli.cpp cgtmath.cpp cgtdigest.cpp cgtpool.cpp cgtpkpool.cpp cgtspan.cpp >nul
if errorlevel 1 (echo [!] host compile failed & exit /b 1)

set /a n=0
:loop
set /a n+=1
echo [*] attempt !n!/%MAXR%: compile device obj ...
del /q cgtgpu.obj cgtkey.exe 2>nul
nvcc -O3 -std=c++14 -Xptxas -O3 -DCGT_STRIDE_HALF=%SH% -allow-unsupported-compiler ^
  -gencode=arch=compute_%CC%,code=%ARCH% -c cgtgpu.cu -o cgtgpu.obj >nul 2>nul
if not exist cgtgpu.obj (echo     compile crashed & goto again)
nvcc -allow-unsupported-compiler -gencode=arch=compute_%CC%,code=%ARCH% ^
  cgtcli.obj cgtmath.obj cgtdigest.obj cgtpool.obj cgtpkpool.obj cgtspan.obj cgtgpu.obj ^
  -o cgtkey.exe 2>_link.log
if not exist cgtkey.exe (echo     link failed & goto again)
echo     smoke test ...
"%~dp0cgtkey.exe" -p %PK% -r %RANGE% >_smoke.log 2>&1
findstr /C:"KEY FOUND" _smoke.log >nul
if not errorlevel 1 (echo     smoke OK & goto done)
echo     smoke FAILED ^(crash or key not found^)
:again
if !n! LSS %MAXR% goto loop
echo [!] No working device object after %MAXR% attempts (cicc kept crashing).
echo     Re-run - it is non-deterministic and usually succeeds within a few tries.
exit /b 1

:done
copy /y cgtkey.exe cgtkey_%SH%.exe >nul 2>nul
del /q _smoke.log _link.log 2>nul
echo(
echo [+] BUILD OK: cgtkey.exe  (also copied to cgtkey_%SH%.exe)
echo [+]   arch=%ARCH%  toolset=%TOOLSET%  CUDA=%CUDADIR%
exit /b 0
