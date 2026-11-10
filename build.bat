@echo off
setlocal
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude src\main.cpp src\image.cpp src\lbp.cpp src\random_forest.cpp -lgdiplus -lgdi32 -o spl_rf.exe
if errorlevel 1 exit /b %errorlevel%
echo Built spl_rf.exe
