if [[ "$#" -ne 1 || ( "$1" != "script") ]]; then
    echo "Usage: ./setup.sh mode"
    echo "mode: script (package mode is not supported for now)"
    echo "package: build and install as a pip package"
    echo "script: build and use as a script. Must be present in local directory for import"
    exit 1
fi

echo "Add -gencode to match all the GPU architectures you have."
echo "Check 'https://en.wikipedia.org/wiki/CUDA#GPUs_supported' for list of architecture."
echo "Check 'http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html' for GPU compilation based on architecture."

# GPU architecture short list:
# GTX 650M: 30
# GTX Titan: 35
# GTX Titan Black: 35
# Tesla K40c: 35
# GTX Titan X: 52
# Titan X (Pascal): 61
# GTX 1080: 61
# GTX 2080: 75
# Titan Xp: 61
# Titan V: 70

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
HAS_CUDA=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$HAS_CUDA" == "True" ]; then
    if ! type nvcc >/dev/null 2>&1 ; then
        echo 'cuda available but nvcc not found. Please add nvcc to $PATH. '
        exit 1
    fi
    cd src
    HERE=$(pwd -P)
    cmd="nvcc -c -std=c++11 -o nnd_cuda.cu.o nnd_cuda.cu -x cu -Xcompiler -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC -I ${HERE} -I ${TORCH}/lib/include\
        -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_75,code=sm_75 \
        -gencode arch=compute_61,code=sm_61"
    echo "$cmd"
    eval "$cmd"
    cd ..
fi

if [ "$1" = "package" ]; then
    # for install
    python setup.py install
elif [ "$1" = "script" ]; then
    # for build
    python build.py
else
    echo "Shouldn't be here."
fi
