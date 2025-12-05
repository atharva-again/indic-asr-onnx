from setuptools import setup, find_packages
import os

# Read version
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), 'indic-asr-onnx', '_version.py')
    with open(version_file, 'r') as f:
        content = f.read()
    local_vars = {}
    exec(content, {}, local_vars)
    return local_vars['__version__']

# Read requirements
def read_requirements():
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(req_file, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="indic-asr-onnx",
    version=read_version(),
    packages=['indic-asr-onnx'],
    install_requires=read_requirements(),
    author="Atharva Verma",
    author_email="atharva.verma18@gmail.com",
    description="Quantized IndicConformer ASR for multiple Indian languages",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/atharva-again/indic-asr-onnx",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    keywords="asr speech-recognition indic-languages multilingual conformer onnx quantized",
)