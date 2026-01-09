export HECATE=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )
export CC="clang"
export CXX="clang++"

alias hopt=$HECATE/build/bin/hecate-opt
alias hopt-debug=$HECATE/build-debug/bin/hecate-opt

mkdir -p $HECATE/examples/traced
mkdir -p $HECATE/examples/traced/cst
mkdir -p $HECATE/examples/benchinputs

build-hopt()(
cd $HECATE/build
ninja
)

build-hoptd()(
cd $HECATE/build-debug
ninja
)

archived-trace()(
    cd $HECATE/examples
    echo "Command: python3 $HECATE/examples/archived/benchmarks/$1.py ${@:3}"
    python3 $HECATE/examples/archived/benchmarks/$1.py ${@:3}
)

archived-test()(
    cd $HECATE/examples
    echo "Command: python3 $HECATE/examples/archived/tests/$3.py $1 $2 ${@:4}"
    python3 $HECATE/examples/archived/tests/$3.py $1 $2 ${@:4}
)

hc-trace()(
    # check if --opt is in the arguments
    has_opt=0
    for arg in "$@"; do
        if [ "$arg" = "--opt" ]; then
            has_opt=1
            break
        fi
    done
    # if --opt is in the arguments, run with new command
    if [ "$has_opt" -eq 1 ]; then
        echo "Detected --opt > running with new command"
    else
        echo "No --opt > running with archived command"
        archived-trace ${@:1}
        exit 1
    fi

    cd $HECATE/examples
    found_file=$(find $HECATE/examples/benchmarks -mindepth 2 -name "$1.py" -type f | head -n 1)
    if [ -z "$found_file" ]; then
        echo "File not found"
        exit 1
    fi
        echo "File found: $found_file"
        echo "Command: python3 $found_file ${@:1}"
        echo "================================================"
        python3 $found_file ${@:1}
)

hc-opt()(
    # check if --opt is in the arguments
    has_opt=0
    for arg in "$@"; do
        if [ "$arg" = "--opt" ]; then
            has_opt=1
            break
        fi
    done
    # if --opt is in the arguments, run with new command
    if [ "$has_opt" -eq 1 ]; then
        echo "Detected --opt > running with new command"
    else
        echo "No --opt > running with archived command"
        hbt ${@:1}
        exit 0
    fi

    cd $HECATE/examples
    echo "Command: python3 $HECATE/examples/hc_opt.py ${@:1}"
    python3 $HECATE/examples/hc_opt.py ${@:1}
    echo "======================================"
)

hc-test()(
    # check if --opt is in the arguments
    has_opt=0
    for arg in "$@"; do
        if [ "$arg" = "--opt" ]; then
            has_opt=1
            break
        fi
    done
    # if --opt is in the arguments, run with new command
    if [ "$has_opt" -eq 1 ]; then
        echo "Detected --opt > running with new command"
    else
        echo "No --opt > running with archived command"
        archived-test ${@:1}
        exit 0
    fi

    cd $HECATE/examples
    found_file=$(find $HECATE/examples/tests -mindepth 2 -name "$1.py" -type f | head -n 1)
    if [ -z "$found_file" ]; then
        echo "File not found"
        exit 1
    fi
        echo "File found: $found_file"
        echo "Command: python3 $found_file ${@:1}"
        echo "======================================"
        python3 "$found_file" "${@:1}"
)

_hc_print_time_table() {
    local t0=$1
    local t1=$2
    local t2=$3
    local t3=$4

    python3 -c "
import sys
try:
    t0, t1, t2, t3 = map(float, sys.argv[1:])
    d_trace = t1 - t0
    d_opt   = t2 - t1
    d_test  = t3 - t2
    d_total = t3 - t0
    d_trace_opt = d_trace + d_opt

    print('\n' + '='*32)
    print(f'| {\"METRIC\":<15} | {\"TIME (sec)\":>10} |')
    print('|' + '-'*30 + '|')
    print(f'| {\"hc-trace\":<15} | {d_trace:>10.5f} |')
    print(f'| {\"hc-opt\":<15} | {d_opt:>10.5f} |')
    print(f'| {\"hc-test\":<15} | {d_test:>10.5f} |')
    print('|' + '-'*30 + '|')
    print(f'| {\"Trace + Opt\":<15} | \033[1;32m{d_trace_opt:>10.5f}\033[0m |')
    print(f'| {\"Total Runtime\":<15} | \033[1;36m{d_total:>10.5f}\033[0m |')
except ValueError:
    print('Error: Invalid time data')
" "$t0" "$t1" "$t2" "$t3"
}

hc-tot()(
    cd $HECATE/examples
    total_time=$(date +%s.%N)
    echo "======================================"
    echo "Starting hc-trace ${@:1}"
    hc-trace ${@:1}
    trace_time=$(date +%s.%N)
    echo "======================================"
    echo "Starting hc-opt ${@:1}"
    hc-opt ${@:1}
    opt_time=$(date +%s.%N)
    echo "======================================"
    echo "Starting hc-test ${@:1}"
    hc-test ${@:1}
    test_time=$(date +%s.%N)

    _hc_print_time_table "$total_time" "$trace_time" "$opt_time" "$test_time"
    echo "======================================"
    echo "Finished hc-tot ${@:1}"
)

hc-eval()(
    echo "======================================"
    echo "Starting hc-eval ${@:1}"
    # Parse fixed arguments: benchmark --opt optimization
    benchmark="$1"
    optimization="$3"  # --opt is $2, optimization name is $3
    
    # Generate log directory and filename
    if [ -n "$benchmark" ] && [ -n "$optimization" ]; then
        log_dir="$HOME/volume/eval/$benchmark/$optimization"
        mkdir -p "$log_dir"
        log_file="$log_dir/hc-eval_$(date +%m%d_%H%M%S).log"
    else
        log_dir="$HOME/volume/eval"
        mkdir -p "$log_dir"
        if [ $# -eq 0 ]; then
            log_file="$log_dir/hc-eval_$(date +%m%d_%H%M%S).log"
        else
            cmd_args=$(echo "${@}" | tr ' ' '_')
            log_file="$log_dir/hc-eval_${cmd_args}_$(date +%m%d_%H%M%S).log"
        fi
    fi
    
    time_tmp="$log_dir/time_data.tmp"
    source $HECATE/hc_eval.sh
    echo "Log file: $log_file"
    echo "Command execution started: $(date)" | tee "$log_file"
    echo "Executed command: hc-eval ${@}" | tee -a "$log_file"
    echo "======================================" | tee -a "$log_file"
    {
        start_time=$(date +%s.%N)
        hc-trace ${@:1}
        trace_time=$(date +%s.%N)
        hc-opt ${@:1}
        opt_time=$(date +%s.%N)
        hc-test ${@:1} --epochs=10
        test_time=$(date +%s.%N)
        echo "======================================"
        echo "$start_time $trace_time $opt_time $test_time" > "$time_tmp"
    } 2>&1 | tee -a "$log_file"

    
    source $HECATE/eval_print.sh
    {
        hc-test ${@:1} --epochs=1
        echo "======================================"
    } 2>&1 | tee -a "$log_file"

    echo "Finished hc-eval ${@:1}"
    echo "Log saved to: $log_file, $(date)"
    if [ -f "$time_tmp" ]; then
        read -r t0 t1 t2 t3 < "$time_tmp"
        _hc_print_time_table "$t0" "$t1" "$t2" "$t3" | tee -a "$log_file"
        rm "$time_tmp"
    else
        echo "Error: Time data not found." | tee -a "$log_file"
    fi
    echo "======================================"
)

hopt-print(){
hopt --$1 --ckks-config="$HECATE/config.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-timing -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-debug-print(){
hopt-debug --$1 --ckks-config="$HECATE/config.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-disable-threading  --mlir-timing --mlir-print-ir-after-failure -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-debug-print-all(){
hopt-debug --$1 --ckks-config="$HECATE/config.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-disable-threading  --mlir-timing --mlir-print-ir-after-failure --debug -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-timing-only(){
hopt --$1 --ckks-config="$HECATE/config.json" --waterline=$2 $HECATE/examples/traced/$3.mlir --mlir-timing -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

# hopt-silent(){
# hopt --$1 --ckks-config="$HECATE/config.json" --waterline=$2 $HECATE/examples/traced/$3.mlir -o $HECATE/examples/optimized/$1/$3.$2.mlir
# }
hopt-silent(){
hopt --$1 --ckks-config="$HECATE/profiled_$4_$5.json" --waterline=$2 $HECATE/examples/traced/$3.mlir -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hc-opt-test() {
hopt-silent $1 $2 $3 $4 $5 && hc-test $1 $2 $3 $4 $5
}

hc-opt-test-timing() {
hopt-timing-only $1 $2 $3 $4 $5 && hc-test $1 $2 $3 $4 $5
}

hopt-lib-hw() {
hopt --$1 --ckks-config="$HECATE/profiled_$4_$5.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-disable-threading --mlir-timing --mlir-print-ir-after-failure -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-lib-hww() {
hopt --$1 --ckks-config="$HECATE/profiled_$4_$5.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-disable-threading --mlir-timing --mlir-print-ir-after-failure -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-unroll() {
hopt --$1 --ckks-config="$HECATE/config.json" --waterline=$2 --unroll-factor=$6 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-disable-threading --mlir-timing --mlir-print-ir-after-failure -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hc-back-opt-test(){
hopt-lib-hww $1 $2 $3 $4 $5 && hc-test $1 $2 $3 $4 $5 $6 $7
}

hc-back-opt(){
hopt-lib-hww $1 $2 $3 $4 $5
}

# alias hopts-heaan-cpu=hopt-heaan-cpu
# alias hopts-heaan-gpu=hopt-heaan-gpu
# alias hopts-seal=hopt-seal
alias hbcot=hc-back-opt-test
alias hbt=hc-back-opt
alias hur=hopt-unroll

alias hoptd=hopt-debug-print
alias hopta=hopt-debug-print-all
alias hopts=hopt-silent
alias hoptt=hopt-timing-only
alias hoptp=hopt-print
alias hcot=hc-opt-test
alias hcott=hc-opt-test-timing
