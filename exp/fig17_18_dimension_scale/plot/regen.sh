#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_scal_dim.py      # writes mrl_dimension.{dat,gp} and scalability_100m.{dat,gp}
gnuplot mrl_dimension.gp
gnuplot scalability_100m.gp
