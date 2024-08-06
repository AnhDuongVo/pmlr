# PMLR
263-5051-00L AI Center Projects in Machine Learning Research FS2024

## How to create environment? (using venv)
> Using venv is advisable because conda and miniconda take too much space.
Here is the command that you can run to create the virtual environment. It will save the environment in current directory.
```
python3 -m venv pmlr_env
source pmlr_env/bin/activate
pip3 install pip --upgrade
pip3 install -U pip wheel setuptools
pip3 install concrete-ml==1.4.1 matplotlib pandas notebook
```
Or you can just run the `setup_venv.sh` script:
```
bash setup_venv.sh
```
### Check if everything works well
First run the `srun` to start the interactive session with GPU:
```
srun -n 2 --mem-per-cpu=4000 -t 60 --account=pmlr-24   --pty bash
```
You can activate the environment with this command:
```
source pmlr_env/bin/activate
```
Then lets run the toyexample. 
```
python toyExample/toyExample.py
```
> After all installations it is advisable to clear the `pip` cache. You can just clear the cache with this command: `pip cache purge`. Remember we don't have much space.
