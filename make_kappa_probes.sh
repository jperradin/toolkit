#!/bin/bash
# Generate kappa_T "probe" runs at chosen (rho,T) state points, two routes:
#   OPTION 1 (NPT): log volume EVERY step -> kappa_T = Var(V)/(<V> kB T).
#   OPTION 2 (NVT): log the CONFIGURATIONAL pressure EVERY step + dump configs
#                   -> kappa_T = 1/[ <Born> - (V/kB T) Var(P_c) ]  (Born from
#                   the SHIK hypervirial, see toolkit/shik.py). Fine (per-step)
#                   pressure sampling is REQUIRED for the variance term.
#
# Each probe starts from an already-equilibrated box_rho{LAB}_{T}K.data, so it
# only needs a short re-equilibration. Both routes are cross-checked against the
# EOS kappa_T (toolkit/llcp_analysis.py) by toolkit/kappa_fluctuations.py.
#
# usage: ./toolkit/make_kappa_probes.sh N  "LAB:T  LAB:T ..."
#   e.g. ./toolkit/make_kappa_probes.sh 1  "215:3000 215:2750 220:3000"
#   (LAB = rho*100; needs sample-N/rho-LAB/box_rhoLAB_TK.data to exist)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LMP=/lus/home/CT9/ccm0504/jperradin/LAMMPS/install/bin/lmp
S="${1:?usage: make_kappa_probes.sh N \"LAB:T ...\"}"; shift
POINTS="${*:?give state points as LAB:T}"
SD="$ROOT/sample-$S"

EQ=50000        # re-equilibration steps (0.08 ns)
PROD=300000     # production steps (0.48 ns) with per-step logging
DUMP=2000       # config dump stride (option 2 Born average)

for pt in $POINTS; do
  LAB="${pt%%:*}"; T="${pt##*:}"
  ISO="$SD/rho-$LAB"
  BOX="$ISO/box_rho${LAB}_${T}K.data"
  if [ ! -f "$BOX" ]; then
    echo "skip $pt: no $BOX" >&2; continue
  fi
  # target pressure (bar) for the NPT route: <P> at this T from the isochore table
  PBAR=$(awk -v t="$T" '$1==t{print $3; exit}' "$ISO/isochore-PT.dat")
  [ -n "$PBAR" ] || { echo "skip $pt: no <P> in isochore-PT.dat" >&2; continue; }
  PROBE="$ISO/kappa-${T}K"; mkdir -p "$PROBE"
  SEED=$((12345 + LAB + T))

  # ---------------- OPTION 1: NPT, volume fluctuations ----------------
cat > "$PROBE/npt.in" <<EOF
units           metal
atom_style      charge
pair_style      shik/wolf 8.0 10.0
read_data       ../box_rho${LAB}_${T}K.data
neighbor        0.3 bin
neigh_modify    every 1 delay 0 check no
timestep        0.0016
velocity        all create $T $SEED mom yes rot yes dist gaussian
# NPT at the isochore <P> so the mean density matches the NVT state point
fix             1 all npt temp $T $T 0.16 iso $PBAR $PBAR 1.6
variable        v equal vol
thermo          1000
thermo_style    custom step temp press vol density pe
run             100000                       # equilibrate the volume
fix             vl all ave/time 1 1 1 v_v file vol.dat   # volume EVERY step
run             $PROD                         # production
EOF

  # ---------------- OPTION 2: NVT, configurational-pressure fluctuations ----
cat > "$PROBE/nvt_fine.in" <<EOF
units           metal
atom_style      charge
pair_style      shik/wolf 8.0 10.0
read_data       ../box_rho${LAB}_${T}K.data
neighbor        0.3 bin
neigh_modify    every 1 delay 0 check no
timestep        0.0016
velocity        all create $T $SEED mom yes rot yes dist gaussian
fix             1 all nvt temp $T $T 0.16
# configurational (virial-only) pressure -> matches the SHIK Born/hypervirial
compute         pc all pressure NULL virial
variable        pc equal c_pc
thermo          1000
thermo_style    custom step temp press c_pc pe
run             $EQ                           # re-equilibrate
fix             pl all ave/time 1 1 1 v_pc file pc.dat    # P_c EVERY step
dump            dd all custom $DUMP conf.dump id type x y z
dump_modify     dd sort id
run             $PROD                         # production
EOF

  { echo "#!/bin/bash"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --ntasks-per-node=192"
    echo "#SBATCH --time=04:00:00"
    echo "#SBATCH --job-name=kap-s$S-$LAB-$T"
    echo "#SBATCH --output=out-%x.o%j"
    echo "module purge; module load PrgEnv-intel/8.6.0"
    echo "srun $LMP -i npt.in      -log log.npt"
    echo "srun $LMP -i nvt_fine.in -log log.nvt"
  } > "$PROBE/start_lammps.sh"
  chmod +x "$PROBE/start_lammps.sh"
  echo "made $PROBE  (T=$T K, <P>=$PBAR bar)"
done
