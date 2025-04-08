import subprocess
import sys
import os

# Get absolute path to current directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Core dependencies to install first
pip_dependencies = [
    'numpy==1.26.4',
    'torch==2.1.1',
    'torchvision==0.16.1',
    'torchaudio==2.1.1',
    'openmim==0.3.8',
    'xformers==0.0.23',
    'fairscale',
    'jupyter',
    'more-itertools',
    'openpyxl',
    'scikit-image',
    'ftfy',
    'cellpose',
    'mmpretrain',
]

# MIM dependencies
mim_dependencies = [
    'mmengine==0.10.4',
    'mmcv==2.1.0',
]

def install():
    print('=== StomataPy Installation ===')
    
    # Step 1: Install pip dependencies
    print('Installing pip dependencies...')
    for pkg in pip_dependencies:
        print(f'Installing {pkg}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
    
    # Step 2: Install MIM dependencies
    print('\nInstalling MIM dependencies...')
    for pkg in mim_dependencies:
        print(f'Installing {pkg}')
        subprocess.check_call([sys.executable, '-m', 'mim', 'install', pkg])
    
    # Step 3: Install submodules with absolute paths
    submodules = ['mmdetection', 'mmsegmentation', 'sam-hq']
    print('\nInstalling submodules...')
    
    for submodule in submodules:
        submodule_path = os.path.join(current_dir, submodule)
        if os.path.exists(submodule_path):
            abs_path = os.path.abspath(submodule_path)
            print(f'Installing {submodule} from {abs_path}')
            # Note: -v and -e are separate arguments
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-v', '-e', abs_path])
        else:
            print(f'Error: Submodule directory not found: {submodule_path}')
    
    print('\n=== Installation Complete ===')
    print('Try importing the modules to verify installation:')
    print('from mmdet.utils import register_all_modules')
    print('from mmseg.utils import register_all_modules')
    print('import segment_anything')

if __name__ == '__main__':
    install()