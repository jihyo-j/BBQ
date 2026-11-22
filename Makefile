# BBQ artifact — top-level entry points.  `source exp/env.sh` first for anything that builds BBQ.
.PHONY: all bbq test smoke verify plots workspace clean

all: verify

bbq:                 ## build build/bbq_hnsw
	./bbq/build.sh search

test:                ## AVX-512 sign-packing unit test
	./bbq/build.sh test && ./build/test_adcbq_sign_packing

smoke: bbq           ## ~1 min end-to-end run on synthetic data (build, search, calibrate, serve)
	./exp/smoke/run.sh

build/verify: exp/verify.cpp exp/common/verify.h
	@mkdir -p build && $(CXX) -std=c++17 -O2 -Wall -Wextra -o $@ exp/verify.cpp

verify: build/verify ## check every paper number in exp/claims.csv against exp/*/results
	./build/verify --root exp

plots:               ## re-draw every figure from exp/*/results (gnuplot >= 5.4)
	@for f in exp/fig*/plot/regen.sh; do echo "== $$f"; $$f > /dev/null || exit 1; done

workspace:           ## lay out exp/experiment/ the way the original runners expect, and build their binaries
	./exp/tools/setup_workspace.sh

clean:
	rm -rf build exp/experiment
