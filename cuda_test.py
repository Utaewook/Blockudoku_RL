import torch

if __name__ == "__main__":
    print(f"CUDA is available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.get_device_name(0)}")\
            

