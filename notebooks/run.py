import papermill as pm
from pathlib import Path

current_directory = Path('.')
files = [file for file in current_directory.rglob('*.ipynb')]

for file in files:
    print(f'Executing : {file}')
    pm.execute_notebook(
        input_path=file,
        output_path=file
    )
