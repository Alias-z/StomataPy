#!/usr/bin/env python3
"""Universal UV installation for StomataPy (works on local machines and Colab)"""
import subprocess
import os


def run_cmd(cmd, allow_fail=False):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if allow_fail:
            print(f"⚠️ Command failed but continuing: {e}")
            return False
        else:
            raise


def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    print("=== StomataPy UV Setup ===")
    
    if is_colab():
        print("🔬 Detected Google Colab environment")
    else:
        print("💻 Detected local environment")
    
    # Install numpy first to lock the version
    print("Installing numpy 1.26.4 first...")
    run_cmd(['uv', 'pip', 'install', 'numpy==1.26.4'])
    
    # Install build tools
    print("Installing build tools...")
    run_cmd(['uv', 'pip', 'install', 'hatchling', 'setuptools', 'wheel', 'editables'])
    
    # Install PyTorch GPU with numpy already locked
    print("Installing PyTorch with GPU support...")
    run_cmd(['uv', 'pip', 'install', 'torch==2.1.1', 'torchvision==0.16.1', 'torchaudio==2.1.1', 
             '--index-url', 'https://download.pytorch.org/whl/cu121'])
    
    # Install other dependencies
    print("Installing other dependencies...")
    deps = [
        'openmim==0.3.8', 'xformers==0.0.23', 'fairscale',
        'jupyter', 'more-itertools', 'openpyxl', 'scikit-image', 'ftfy',
        'cellpose', 'mmpretrain', 'mmengine==0.10.4', 'mmcv==2.1.0'
    ]
    run_cmd(['uv', 'pip', 'install'] + deps)
    
    # Install submodules in editable mode
    print("Installing submodules...")
    editable_submodules = ['mmdetection', 'mmsegmentation', 'sam-hq']
    for sub in editable_submodules:
        if os.path.exists(sub):
            print(f"Installing {sub}...")
            success = run_cmd(['uv', 'pip', 'install', '-e', f'./{sub}', '--no-build-isolation'], allow_fail=True)
            if not success:
                print(f"⚠️ Failed to install {sub}, skipping...")
    
    # Install SAHI separately (not in editable mode due to build issues)
    if os.path.exists('sahi'):
        print("Installing SAHI...")
        success = run_cmd(['uv', 'pip', 'install', './sahi'], allow_fail=True)
        if not success:
            print("⚠️ Failed to install local SAHI, installing from PyPI...")
            run_cmd(['uv', 'pip', 'install', 'sahi'], allow_fail=True)
    
    print("\n✅ Installation complete!")
    
    if is_colab():
        print("Use: import torch; torch.cuda.is_available()")
    else:
        print("Use: .venv\\Scripts\\python your_script.py")
        print("Or:  .venv\\Scripts\\activate && python your_script.py")


if __name__ == '__main__':
    main() 