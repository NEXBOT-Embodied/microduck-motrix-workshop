# Windows entry point for the shared workshop installer.
$ErrorActionPreference = 'Stop'
$git = Get-Command git.exe -CommandType Application -ErrorAction SilentlyContinue
$bash = if ($git) {
    $git.Source -replace '\\(?:cmd|bin|mingw64\\bin)\\git\.exe$', '\bin\bash.exe'
}
if (!$bash -or $bash -eq $git.Source -or !(Test-Path -LiteralPath $bash -PathType Leaf)) {
    throw 'Install Git for Windows and make git.exe available on PATH.'
}

Push-Location -LiteralPath $PSScriptRoot
try {
    & $bash ./install.sh @args
    $status = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $status
