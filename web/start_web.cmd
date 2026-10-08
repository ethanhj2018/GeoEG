@echo off
rem Double-click to start the GeoEG demo page in WSL, then open http://localhost:7860
wsl -d Ubuntu-22.04 --cd "%~dp0" -- bash start_web.sh
pause
