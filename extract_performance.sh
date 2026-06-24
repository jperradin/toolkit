#!/bin/bash
# Extract LAMMPS performance from every rho-*/log.lammps of a sample.
# Each `run` writes a pair of lines:
#   Loop time of <wall_s> on <procs> procs for <steps> steps with <atoms> atoms
#   Performance: <ns/day> ns/day, <h/ns> hours/ns, <steps/s> timesteps/s
# Per log we report: #runs, mean/min/max ns/day, total wall (h), CPU-h
# (= wall * procs), total steps.
#
# usage: ./toolkit/extract_performance.sh [N]      (sample number, default 1)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S="${1:-1}"
SD="$ROOT/sample-$S"
[ -d "$SD" ] || { echo "no such sample dir: $SD" >&2; exit 1; }

printf "%-10s %5s %10s %10s %10s %12s %12s %14s\n" \
       "rho" "runs" "ns/day" "min" "max" "wall(h)" "cpu(h)" "steps"

tot_runs=0; tot_wall=0; tot_cpu=0; tot_steps=0
for d in "$SD"/rho-*/; do
  lab="$(basename "$d")"
  case "$lab" in *_test) continue;; esac      # skip scratch/test dirs
  # one log.lammps, else the chained per-segment logs (concatenated in order)
  if [ -f "$d/log.lammps" ]; then
    logs=("$d/log.lammps")
  else
    logs=( $(ls "$d"/log.seg*.lammps 2>/dev/null | sort) )
  fi
  [ "${#logs[@]}" -gt 0 ] || { printf "%-10s %5s\n" "$lab" "nolog"; continue; }

  read -r runs mean mn mx wall cpu steps < <(cat "${logs[@]}" | awk '
    /Loop time of/        { wall += $4; steps += $9; procs = $6 }
    /Performance:/        { nd=$2; sum+=nd; n++;
                            if (n==1||nd<mn) mn=nd;
                            if (n==1||nd>mx) mx=nd }
    END { if (n==0) { print "0 0 0 0 " wall/3600 " " wall/3600*procs " " steps }
          else printf "%d %.3f %.3f %.3f %.1f %.1f %d\n",
                      n, sum/n, mn, mx, wall/3600, wall/3600*procs, steps }
  ')

  printf "%-10s %5s %10s %10s %10s %12s %12s %14s\n" \
         "$lab" "$runs" "$mean" "$mn" "$mx" "$wall" "$cpu" "$steps"

  tot_runs=$((tot_runs + runs))
  tot_wall=$(awk -v a="$tot_wall" -v b="$wall" 'BEGIN{printf "%.1f", a+b}')
  tot_cpu=$(awk -v a="$tot_cpu" -v b="$cpu" 'BEGIN{printf "%.1f", a+b}')
  tot_steps=$((tot_steps + steps))
done

echo "-----------------------------------------------------------------------------------------------"
printf "%-10s %5s %10s %10s %10s %12s %12s %14s\n" \
       "TOTAL" "$tot_runs" "-" "-" "-" "$tot_wall" "$tot_cpu" "$tot_steps"
