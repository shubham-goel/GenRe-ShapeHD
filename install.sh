# conda create -n shaperecon_cuda102 python=3.6
# conda activate shaperecon_cuda102

# export CUDA_HOME=/usr/local/cuda-10.2
# export PATH=/usr/local/cuda-10.2/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH

# conda install -c pytorch torchvision=0.2.1 cudatoolkit=10.2
# conda install -c conda-forge \
#     pandas=0.23.4 \
#     tqdm=4.28.1 \
#     scikit-image=0.14.0 \
#     numba=0.41.0 \
#     opencv=3.4.2 \
#     tensorflow=1.5.1 \
#     trimesh=2.35.47 \
#     rtree=0.8.3 \
#     scikit-learn=0.20.1 


######### This installation needs to be done only once - START ###########

# Clone Genre, or copy from /home/shubham/code/GenRe-ShapeHD
git clone git@github.com:shubham-goel/GenRe-ShapeHD.git $GENRE_ROOT
cd $GENRE_ROOT
GENRE_ROOT=`pwd`
GENRE_ROOT=/home/shubham/code/GenRe-ShapeHD

# Using singularity container for installing pytorch0.4.1 w/ cuda10
sandbox=/home/$USER/sandbox/genre2
singularity build --fakeroot --sandbox $sandbox docker://nvcr.io/nvidia/pytorch:18.09-py3
singularity shell --nv --fakeroot --writable $sandbox
mkdir /genre
exit

# Get gpu
srun --gres=gpu:1 --pty bash

# Open singularity
singularity shell --nv --fakeroot \
    --bind $GENRE_ROOT:/genre \
    --writable $sandbox
cd /genre

# pytorch0.4.1 + cuda10.0 should be available already
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available());'

# install genre dependencies
pip install --upgrade pip
pip install \
    scikit-image==0.14.0 \
    opencv-python==3.4.2.17 \
    tensorflow==1.5.1 \
    rtree==0.8.3 \
    trimesh[all]==2.35.47 \
    scikit-learn==0.20.1 
pip install llvmlite==0.32.1 numba==0.41.0

# install genre cuda kernels
./clean_toolbox_build.sh
./build_toolbox.sh

# Exit singularity
exit
# release gpu
exit

# Test
srun --gres=gpu:1 singularity exec --nv \
    --bind $GENRE_ROOT:/genre \
     $sandbox bash -c "cd /genre && ./scripts/test_genre.sh 0"

# Save an image to run quickly
singularity build --fakeroot ~/genre.sif $sandbox
######### This installation needs to be done only once - END ###########

GENRE_ROOT=/home/shubham/code/GenRe-ShapeHD
srun --gres=gpu:1 singularity exec --nv \
    --bind $GENRE_ROOT:/genre \
     ~/genre.sif bash -c "cd /genre && ./scripts/test_genre.sh 0"
