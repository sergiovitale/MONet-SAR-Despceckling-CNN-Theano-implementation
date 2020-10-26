export PATH=/usr/lib/nvidia-cuda-toolkit/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/nvidia-cuda-toolkit/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/lib/nvidia-cuda-toolkit/cuda/include:$CPATH
export LIBRARY_PATH=/usr/lib/nvidia-cuda-toolkit/cuda/lib64:$LIBRARY_PATH
export THEANO_FLAGS='cuda.root=//usr/lib/nvidia-cuda-toolkit/bin,device=cuda0,floatX=float32,init_gpu_device=cuda0'
spyder
