import numpy as np 
import sys,os

from natsort import natsorted

rootdir = os.getcwd()
n_atoms = np.int32(sys.argv[1])


boxes = np.zeros((120))
try: # try to find boxes files if not raise error
    with open("../../analysis/boxes",'r') as inp:
        for li,l in enumerate(inp):
            boxes[li] = float(l)
except:
    raise ValueError 

counter_file = 0
for subdir, dirs, files in os.walk(rootdir):
    dirs.sort()
    files = natsorted(files)
    for file in files:
        if file.split('.')[1] == "data":
            with open(file,'r') as inp:
                print(file)
                with open(file.split('.')[0]+'.xyz','w') as out:
                    counter_line = 0
                    for li,l in enumerate(inp):
                        try:
                            if l.split('.')[0] == " Atoms":
                                out.write(f'Lattice=\"{boxes[counter_file]} 0.0 0.0 0.0 {boxes[counter_file]} 0.0 0.0 0.0 {boxes[counter_file]}\"\n')
                                counter_line += 1
                            elif l.split()[0] == f'{n_atoms}':
                                out.write(l)
                            elif l.split()[0] == "1":
                                parts = l.split()
                                out.write(f'o {parts[1]} {parts[2]} {parts[3]}\n')
                            elif l.split()[0] == "2":
                                parts = l.split()
                                out.write(f'si {parts[1]} {parts[2]} {parts[3]}\n')
                            elif l.split()[0] == "3":
                                parts = l.split()
                                out.write(f'Na {parts[1]} {parts[2]} {parts[3]}\n')
                        except:
                            pass
            counter_file += 1
