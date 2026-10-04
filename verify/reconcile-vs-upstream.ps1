<# 复算：本树 vs 上游基线，逐文件 SHA256。默认路径按本仓结构。 #>
param(
  [string]$Baseline = 'refs\infer-all-full\infer-all-master',
  [string]$Tree     = '.\src-tree\fusion-engine-src'
)
$ErrorActionPreference = 'Stop'
function Map-Tree($root){
  $m = @{}
  Get-ChildItem $root -Recurse -File -Force |
    Where-Object { $_.FullName -notmatch '\\__pycache__\\' -and $_.Extension -ne '.pyc' } |
    ForEach-Object { $m[$_.FullName.Substring($root.Length + 1)] = $_.FullName }
  return $m
}
$b = Map-Tree $Baseline
$a = Map-Tree $Tree
$same = 0; $mod = @()
foreach ($k in $b.Keys) {
  if ($a.ContainsKey($k)) {
    if ((Get-FileHash $b[$k] -Algorithm SHA256).Hash -eq (Get-FileHash $a[$k] -Algorithm SHA256).Hash) { $same++ }
    else { $mod += $k }
  }
}
$added = @($a.Keys | Where-Object { -not $b.ContainsKey($_) })
$miss  = @($b.Keys | Where-Object { -not $a.ContainsKey($_) })
"baseline files = $($b.Count)"
"tree files     = $($a.Count)"
"identical      = $same"
"modified       = $($mod.Count)"
"added          = $($added.Count)"
"missing        = $($miss.Count)"
