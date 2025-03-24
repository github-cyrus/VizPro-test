# Start the Insights Prediction Model
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'Useful insights predicition model'; python app.py"

# Wait for 5 seconds
Start-Sleep -Seconds 5

# Start the Data Cleaning Model
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'Data Cleaning Model\Data Cleaning Model'; python app.py" 