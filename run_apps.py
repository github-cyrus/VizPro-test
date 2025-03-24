import subprocess
import sys
import os
from threading import Thread

def run_insights_prediction():
    insights_dir = os.path.join(os.getcwd(), "Useful insights predicition model")
    if os.path.exists(insights_dir):
        os.chdir(insights_dir)
        subprocess.run([sys.executable, "app.py"])
    else:
        print(f"Error: Directory not found - {insights_dir}")

def run_data_cleaning():
    cleaning_dir = os.path.join(os.getcwd(), "Data Cleaning Model", "Data Cleaning Model")
    if os.path.exists(cleaning_dir):
        os.chdir(cleaning_dir)
        subprocess.run([sys.executable, "app.py"])
    else:
        print(f"Error: Directory not found - {cleaning_dir}")

if __name__ == "__main__":
    # Store the original directory
    original_dir = os.getcwd()
    
    # Create threads for each application
    insights_thread = Thread(target=run_insights_prediction)
    cleaning_thread = Thread(target=run_data_cleaning)
    
    # Start both applications
    insights_thread.start()
    
    # Return to original directory before starting second app
    os.chdir(original_dir)
    cleaning_thread.start()
    
    # Wait for both to complete (they'll run indefinitely)
    insights_thread.join()
    cleaning_thread.join() 