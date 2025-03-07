# Toolkit 

That repository stores all my useful scripts on different branches.

Arborescence:

- main:
  - README.md
- lammps :
  - fetch_thermo.py
  - calculate_densities.py
  - convert_to_ExtendedXYZ.py
  - make_tar-gz.py
- nexus-parallel :
  - launch_nexus.py
  - start_nexus-MUSE.sh
  - start_nexus-ADASTRA.sh
  - get_nexus_results.py
- nexus-serial :
  - launch_nexus.py
  - get_nexus_results.py
- reve-parallel :
  - launch_reve.py
  - start_reve-MUSE.sh
  - start_reve-ADASTRA.sh
  - get_reve_results.py
- reve-serial :
  - launch_reve.py
  - get_reve_results.py

## To pull a branch
```bash
git clone -b <branch> git@github.com:jperradin/toolkit.git
```

## To push on a branch

```bash
git add <modified_scripts> 
git commit -m "updated <modified_scripts>"
git push origin <branch>
```

