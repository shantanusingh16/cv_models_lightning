import subprocess
import sys
from setuptools import setup, find_packages

def get_torch_version():
    try:
        subprocess.check_output('nvidia-smi')
        cuda_available = True
    except Exception: # this command not being found can raise quite a few different errors depending on the configuration
        cuda_available = False
    
    mps_available = sys.platform == "darwin"

    if cuda_available:
        return f"torch==1.13.1+cu117 torchvision==0.14.1+cu117"
    elif mps_available:
        return "torch==1.13.1 torchvision==0.14.1"
    else:
        return "torch==1.13.1 torchvision==0.14.1"

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Remove any existing torch and torchvision from requirements
requirements = [req for req in requirements if not req.startswith(('torch', 'torchvision'))]

# Add the appropriate torch and torchvision versions
requirements.extend(get_torch_version().split())

setup(
    name='cv_models_lightning',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8,<3.9',
    dependency_links=['https://download.pytorch.org/whl/cu117']
)