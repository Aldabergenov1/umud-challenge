"""Execute a notebook cell by cell, saving after every cell so progress is visible."""
import sys, time
import nbformat
from nbclient import NotebookClient

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=None, kernel_name="umud", resources={"metadata": {"path": sys.argv[2]}})
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        t0 = time.time()
        client.execute_cell(cell, i)
        nbformat.write(nb, path)
        print(f"cell {i} done in {time.time() - t0:.0f}s", flush=True)
print("NOTEBOOK DONE", flush=True)
