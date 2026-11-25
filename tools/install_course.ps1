# Windows wrapper. Git Bash is supplied by Git for Windows.
[CmdletBinding()]
param(
    [ValidateSet('cuda', 'cpu')][string]$Backend = 'cuda',
    [switch]$Upstream
)
$ErrorActionPreference = 'Stop'
$git = (Get-Command git -ErrorAction Stop).Source
$bash = $git -replace '\\cmd\\git\.exe$', '\bin\bash.exe' -replace '\\mingw64\\bin\\git\.exe$', '\bin\bash.exe'
if ($bash -eq $git -or -not (Test-Path -LiteralPath $bash)) {
    throw 'Git Bash was not found. Install Git for Windows and retry.'
}
$options = @('--backend', $Backend)
if ($Upstream) { $options += '--upstream' }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $bash ./install_course.sh @options
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
