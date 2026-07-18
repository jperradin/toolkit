#!/bin/bash

tar --use-compress-program=pigz -cvf A-part.tar.gz A-part/
tar --use-compress-program=pigz -cvf B-part.tar.gz B-part/
tar --use-compress-program=pigz -cvf all-boxes.tar.gz all-boxes/
tar --use-compress-program=pigz -cvf analysis.tar.gz analysis/
tar --use-compress-program=pigz -cvf thermo.tar.gz thermo/ log.lammps out-loading*
# you may change the script filenames
tar --use-compress-program=pigz -cvf scripts.tar.gz *.py *.sh chgbox* loading*
