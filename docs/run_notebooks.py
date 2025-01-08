import papermill as pm
from pathlib import Path
import subprocess

def is_git_ignored(file_path):
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(file_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0

print("########################################")
print("#          Notebook Executor           #")
print("########################################")

current_directory = Path(".")
files = [file for file in current_directory.rglob("*.ipynb")]

# filter out git-ignored files
filtered_files = [file for file in files if not is_git_ignored(file)]

print("\n")
print("Files to execute:\n")
print(filtered_files)

for file in filtered_files:
    # print("\n")
    print(f"Executing: {file}")
    pm.execute_notebook(input_path=file, output_path=file, progress_bar=False)
    print(f"Finished: {file}")
    print("\n")

print("All notebooks have been executed.")
