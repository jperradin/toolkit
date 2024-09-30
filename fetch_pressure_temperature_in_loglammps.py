import numpy as np
import os,sys,re
with open(sys.argv[1],'r') as inp:
    checkpoint = False
    pressure = []
    temperature = []
    for li,l in (enumerate(inp)):
        try:
            if l.split()[0] == '31275': #skipping one value but np
                checkpoint = True
                block_pressure = []
                block_temperature = []
        except:
            pass
        if checkpoint: 
            block_pressure.append(float(l.split()[6])*0.0001)
            block_temperature.append(float(l.split()[1]))
        try:
            if l.split()[0] == '62500':
                checkpoint = False
                block_pressure = np.array(block_pressure)
                block_temperature = np.array(block_temperature)
                pressure.append(np.mean(block_pressure))
                temperature.append(np.mean(block_temperature))
        except:
            pass

for i in range(len(pressure)):
    print(f"{pressure[i]:2.2f}\t{temperature[i]:2.2f}")
