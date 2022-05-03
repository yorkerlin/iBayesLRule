% startup script
disp('executing startup script...')
addpath(genpath(pwd))

%Add the python2 path.
pcPythonExe = '/home/yorkerlin/miniconda3/envs/py2/bin/python';
[ver, exec, loaded] = pyversion(pcPythonExe); pyversion
