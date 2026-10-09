# 用 WMI 起隔离引擎（本机铁律：常驻服务只能用 WMI / wscript，从 agent shell 起的会被连带杀掉）
# 参数由 _kvpool_ab_start.py 写好的 cmdline 文件提供，本脚本只负责"发射"。
param(
  [Parameter(Mandatory=$true)][string]$CmdlineFile,
  [Parameter(Mandatory=$true)][string]$LogFile
)

$cmd = (Get-Content -LiteralPath $CmdlineFile -Raw -Encoding UTF8).Trim()
"CMD = $cmd"

# 清日志（本次运行独立成篇）
if (Test-Path -LiteralPath $LogFile) { Remove-Item -LiteralPath $LogFile -Force }

$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
"ReturnValue = $($r.ReturnValue)  ProcessId = $($r.ProcessId)"
