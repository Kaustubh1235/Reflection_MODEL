import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("✅ Success! PyTorch can see your GPU.")
    # Print the CUDA version PyTorch was built with
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    # Get the number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")
    # Print the name of the GPU
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("❌ Failed. PyTorch cannot see your GPU. Please check the installation steps.")