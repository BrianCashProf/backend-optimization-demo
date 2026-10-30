# Running the optimization notebook

Start with [optimization_demo.ipynb](optimization_demo.ipynb). It contains the walkthrough;
the other files support its examples. These instructions run the CPU examples on
**Mac, Linux, or Windows**, including the GIL comparison. No NVIDIA GPU or CUDA
installation is needed. GPU cells will be skipped, including on Apple Silicon Macs.

## 1. Install uv

[uv](https://docs.astral.sh/uv/getting-started/installation/) downloads Python and
creates the project's environments, so you do not need to install Python separately.
You need internet access for the initial setup.

**Mac or Linux — open Terminal:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows — open PowerShell:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal after installation. If uv is already installed,
skip this step.

## 2. Create the environments

Keep all the shared files together in `oct-29th-optimizations`. In your terminal,
enter `cd ` followed by the folder's full path in quotes, for example:

```text
cd "/path/to/oct-29th-optimizations"
```

Then run these same commands on **Mac, Linux, or Windows**:

```text
uv venv --python 3.14 .venv
uv pip install --python .venv -r requirements.txt
uv venv --python 3.14t .venv-free-threaded
```

The first environment runs Jupyter and the examples. The second supplies the special
Python build used by the GIL comparison; it needs no additional packages. uv downloads
both Python versions as needed. See its [environment setup documentation](https://docs.astral.sh/uv/pip/environments/).

These folders are created on your own computer. Recreate them if you move the project
or share it with someone else. The requirements use version ranges, so installed
package versions may differ slightly from the original machine.

## 3. Run the notebook in VS Code

1. Install [VS Code](https://code.visualstudio.com/) and Microsoft's
   [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
   and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
   extensions.
2. Choose **File → Open Folder** and open `oct-29th-optimizations` itself.
   You can use **Terminal → New Terminal** for the setup commands in step 2.
3. Open `optimization_demo.ipynb` from the Explorer.
4. Click **Select Kernel** in the notebook's upper-right corner. Choose
   **Select Another Kernel → Python Environments**, if shown, then select the
   **Python 3.14 environment inside this folder's `.venv`**. Use the standard
   `.venv` for the notebook; it launches the separate free-threaded helper itself.

The selected Python executable should be:

| System | Notebook interpreter |
|---|---|
| Mac or Linux | `.venv/bin/python` |
| Windows | `.venv\Scripts\python.exe` |

If `.venv` is missing from the list, open the Command Palette (**Cmd+Shift+P** on
Mac, **Ctrl+Shift+P** on Windows/Linux), run **Python: Select Interpreter**, and
use **Enter interpreter path** to choose that executable. Then select it in the
notebook's kernel picker. See [VS Code's kernel selection guide](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management).

### Set up the notebook for CPU mode

Add a **code cell at the very top** of the notebook using **+ Code** and paste this
in. It works on all three operating systems and sets the free-threaded Python path
for the notebook running inside VS Code:

```python
import os
from pathlib import Path

os.environ["PRICING_SKIP_CUDA"] = "1"
ft_folder = Path(".venv-free-threaded").resolve()
os.environ["PRICING_FT_PYTHON"] = str(
    ft_folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
)
```

For a quicker first run, change the notebook's `BATCH_SIZE` setting to `200_000`.
Click **Run All** in the notebook toolbar to run every cell from top to bottom,
including the new setup cell. Results and charts appear directly below the cells
in VS Code. You do not need to start JupyterLab in a browser or activate an environment.

The API section starts and stops local servers automatically and creates reports in
`locust-results/`. Open `locust-results/index.html` in your browser to see the saved
reports. A “CUDA skipped” message is expected on this setup.

Save the notebook to keep your setup cell. Next time, reopen this folder and the
notebook, check that `.venv` is selected, and click **Run All**. After restarting
the kernel, run all cells again so the setup cell takes effect.

## What the files are for

| File | Why it is included |
|---|---|
| `optimization_demo.ipynb` | Main notebook to open and run. |
| `requirements.txt` | Packages needed for the CPU notebook, APIs, and load tests. |
| `gil_benchmark.py` | Helper the notebook runs with free-threaded Python for its GIL comparison. |
| `rest_unoptimized.py`, `rest_optimized.py` | REST API examples used by the notebook's performance comparison. |
| `websocket_unoptimized.py`, `websocket_optimized.py` | WebSocket versions of those API examples. |
| `locustfile.py` | Defines the traffic used to measure the APIs. |
| `demo_runner.py` | Starts the APIs, runs the load tests, and saves the reports. |
| `test_demo.py` | Checks that the examples produce consistent results. |
| `requirements-cuda.txt`, `requirements-free-threaded.txt` | Dependency lists for the original GPU environments. Neither is needed for this CPU setup, including the GIL comparison. |
| `register_kernels.py` | Optional helper for the original Linux/WSL custom notebook kernels. Not needed for these instructions. |
| `.gitignore` | Keeps generated environments, caches, and reports out of version control. |

You can leave the supporting scripts as they are; the notebook calls them when needed.
