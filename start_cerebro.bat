@echo off
title Cognitive Core - Life Daemon
echo [SISTEMA] - Iniciando compilacion de liberacion (Release) y lanzando Nucleo Cognitivo...
"%USERPROFILE%\.cargo\bin\cargo" run --release
pause
