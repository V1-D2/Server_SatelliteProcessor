import torch


def get_best_device():
    """Detect and return the best available device"""
    try:
        # 1. Check for CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            return torch.device("cuda"), "CUDA GPU"

        # 2. Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps"), "Apple Silicon GPU (MPS)"

        # 3. Check for DirectML (Intel GPUs, Windows)
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device(), "DirectML (Intel GPU)"
        except ImportError:
            pass

        # 4. Fallback to CPU
        return torch.device("cpu"), "CPU"

    except Exception as e:
        print(f"Error detecting device: {e}")
        return torch.device("cpu"), "CPU (fallback)"