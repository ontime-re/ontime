import papermill as pm
from pathlib import Path

print("########################################")
print("#          Notebook Executor           #")
print("########################################")

current_directory = Path(".")
files = [file for file in current_directory.rglob("*.ipynb")]

print("\n")
print("Files to execute:\n")
print(files)

for file in files:
    #print("\n")
    print(f"Executing: {file}")
    pm.execute_notebook(input_path=file, output_path=file, progress_bar=False)
    print(f"Finished: {file}")
    print("\n")

print("All notebooks have been executed.")
