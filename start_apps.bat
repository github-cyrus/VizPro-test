@echo off
start cmd /k "cd "Useful insights predicition model" && python app.py"
timeout /t 5
start cmd /k "cd "Data Cleaning Model\Data Cleaning Model" && python app.py" 