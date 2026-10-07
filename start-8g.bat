@echo off
rem ============================================================================
rem  start-8g.bat -- 8 GB TIER ONLY. Do NOT use this launcher on other cards.
rem
rem  WHY THIS FILE EXISTS (measured 2026-10-08, this tree):
rem    bf16 KV costs 4.0 MiB per 64-token page; the shipped default pool (2,176
rem    tokens = 34 pages) is ~40x smaller than what an 8 GB card can hold.
rem    With --kv-dtype rk4v4-e8 a page costs ~1.1 MiB, and with --kv-capacity auto
rem    the pool is sized from NINFER_KV_WINDOW (the resident working set), not from
rem    free VRAM: pool = window + prefill_chunk, per lane, plus 8 slack pages
rem    (startup.cpp:134-157, the kvmem_resident_pages port).
rem
rem  RESULTING BUDGET ON 8 GB (%NINFER_KV_WINDOW% below = 65536):
rem    weights PTQ1 5.52 GiB + fixed ~0.50 GiB + KV 1004 pages x 1.1 MiB = ~7.1 GiB
rem    => fits 8 GB with ~0.4-0.5 GiB to spare. A 120k-token prompt still READS IN:
rem    the part beyond the window lives in host RAM and is brought back by retrieval.
rem
rem  ROLLBACK: delete this file. It changes no engine default and no other tier.
rem ============================================================================

setlocal

rem -- the model artifact (override with a path argument or NINFER_MODEL) --
set MODEL=%~1
if "%MODEL%"=="" set MODEL=%NINFER_MODEL%
if "%MODEL%"=="" (
  echo [start-8g] FATAL: no model artifact. Pass it as the first argument or set NINFER_MODEL.
  exit /b 2
)
if not exist "%MODEL%" ( echo [start-8g] FATAL: model not found: %MODEL% & exit /b 2 )

rem -- 8 GB tier switches ------------------------------------------------------------------
rem  WINDOW = the device-resident working set, in tokens. 65536 fits 8 GB with rk4v4-e8;
rem  raise to 98304 only if the card has headroom, lower to 32768 if startup is refused.
if "%NINFER_KV_WINDOW%"=="" set NINFER_KV_WINDOW=65536
set NINFER_KV_RING=1
set NINFER_HOST_PAGEABLE=1
set NINFER_KV_REUSE_HOSTBACKED=1
rem  Do NOT set NINFER_KV_RETRIEVE here: leaving it unset lets the engine pick a retrieval
rem  budget when a request exceeds the pool (B32). An explicit =0 is respected and then a
rem  long prompt answers wrong with HTTP 200 -- that is the failure this tier avoids.
set NINFER_KV_RETRIEVE=
set NINFER_TERNARY_PTQ1_FAST=1

rem -- read these three lines after startup (the configuration is only correct if all hold) --
rem    1) capacity | KV <N> tokens, rk4v4-e8, explicit | pages N/M | runtime X GiB
rem       requirement: X + 5.52 GiB <= the card usable VRAM (7.4-7.6 GiB on 8 GB)
rem    2) [ring] budgets: skeleton=S, evidence=E, recent=R, free=F (pool P)  with S+E+R+F == P
rem    3) the dtype field on line 1 must read rk4v4-e8 (a silently swallowed switch was B22)

echo [start-8g] model = %MODEL%
echo [start-8g] window = %NINFER_KV_WINDOW% tokens, kv-dtype = rk4v4-e8, pool = auto
echo [start-8g] remaining args are passed through: %*

rem --kv-capacity auto + --kv-headroom-mib 1024: leave 1 GiB free; the ring caps the pool
rem at the working set, so auto never spends the whole card on resident KV.
ninfer-serve "%MODEL%" --host 127.0.0.1 --port 8091 --model-id ninfer-local ^
  --kv-capacity auto --kv-headroom-mib 1024 ^
  --kv-dtype rk4v4-e8 ^
  --max-context 131072 --max-concurrency 1 ^
  --prefill-chunk 256 --gdn-state-fp16 --no-cuda-graph ^
  --host-kv-mib 16384
endlocal
