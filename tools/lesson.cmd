@echo off
setlocal
chcp 65001 >nul
rem Use an explicitly selected Python first, then a prepared environment in this package.
if defined COURSE_PYTHON goto python_selected
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "COURSE_PYTHON=%~dp0..\.venv\Scripts\python.exe"
    goto python_selected
)
if exist "%~dp0..\upstream\MotrixLab\.venv\Scripts\python.exe" (
    set "COURSE_PYTHON=%~dp0..\upstream\MotrixLab\.venv\Scripts\python.exe"
    goto python_selected
)
echo Prepared Python not found in this repository.
echo Prepare the course environment first; see README.md.
echo Or set COURSE_PYTHON to an existing prepared Python executable.
exit /b 2
:python_selected
set "PYTHONUTF8=1"
set "MICRODUCK_LESSON_BRIEF=1"
if not exist "%COURSE_PYTHON%" (
    echo Prepared Python not found: %COURSE_PYTHON%
    echo Set COURSE_PYTHON to a prepared Python executable, then run this command again.
    exit /b 2
)
if not exist "%~dp0local_tutorial.py" (
    echo Course entry not found: %~dp0local_tutorial.py
    exit /b 2
)
pushd "%~dp0.." || exit /b 2
set "MICRODUCK_COURSE_ROOT=%CD%"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
"%COURSE_PYTHON%" "%~dp0local_tutorial.py" %*
set "COURSE_EXIT=%ERRORLEVEL%"
popd
exit /b %COURSE_EXIT%
