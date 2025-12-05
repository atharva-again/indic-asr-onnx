# PyPI + Hugging Face Hub Deployment Guide

## Overview
This guide outlines the steps to package the IndicConformerTranscriber as a PyPI-installable package with automatic model downloads from Hugging Face Hub, providing a frictionless user experience.

## Prerequisites
- Python 3.8+
- GitHub account
- Hugging Face account
- PyPI account
- All model files and transcriber.py ready

## Step 1: Prepare Package Structure
1. ~~Create a new directory for the package (e.g., `indic-asr`)~~
2. ~~Set up the following directory structure:~~
   - ~~`indic_asr/` (package directory)~~
     - ~~`__init__.py`~~
     - ~~`transcriber.py`~~
     - ~~`_version.py`~~
   - `tests/` (optional)
   - `docs/` (optional)
   - ~~`setup.py`~~
   - `setup.cfg` (optional)
   - ~~`MANIFEST.in` (optional)~~
   - ~~`README.md`~~
   - ~~`LICENSE`~~
   - ~~`requirements.txt`~~

## Step 2: Create Package Files

### 2.1 ~~setup.py~~
~~Create setup.py with package metadata, dependencies, and entry points.~~

### 2.2 ~~__init__.py~~
~~Define package imports and version.~~

### 2.3 ~~_version.py~~
~~Store version information.~~

### 2.4 ~~README.md~~
~~Write comprehensive documentation including:~~
- ~~Installation instructions~~
- ~~Usage examples~~
- ~~Supported languages~~
- ~~Performance notes~~
- ~~License information~~

### 2.5 ~~LICENSE~~
~~Choose and include appropriate license (e.g., MIT, Apache 2.0).~~

## Step 3: Test Package Locally
1. ~~Install package in development mode: `pip install -e .`~~
2. ~~Test imports and basic functionality~~
3. ~~Verify dependencies are correctly specified~~
4. ~~Run any existing tests~~
5. ~~Build package successfully with uv~~

## Step 4: Prepare Hugging Face Hub Repository
1. ~~Create new repository on Hugging Face Hub~~
2. ~~Set repository type to "Model"~~
3. ~~Configure repository settings:~~
   - ~~Set to public or private as needed~~
   - ~~Add appropriate tags (audio, speech-recognition, multilingual)~~
   - ~~Add model card metadata~~

## Step 5: Upload Model Files to Hugging Face Hub
1. ~~Install Hugging Face CLI: `pip install huggingface_hub[cli]`~~
2. ~~Authenticate: `huggingface-cli login`~~
3. ~~Upload all model files:~~
   - ~~ONNX model files~~
   - ~~Configuration files (vocab.json, language_masks.json)~~
   - ~~transcriber.py (for reference)~~
   - ~~README and documentation~~
4. ~~Create model card with usage examples~~
5. ~~Set up model metadata and tags~~

## Step 6: Update Package for HF Integration
1. ~~Modify transcriber.py to use the correct HF repo ID~~
2. ~~Ensure auto-download functionality works~~
3. ~~Update version numbers if needed~~
4. ~~Test model download and inference~~

## Step 7: Build Distribution Packages
1. ~~Install build tools: `uv pip install build twine`~~
2. ~~Clean previous builds: `rm -rf dist/ build/ *.egg-info/`~~
3. ~~Build source distribution and wheel: `python -m build`~~
4. ~~Verify built packages: `twine check dist/*`~~

## Step 8: Test Package Installation
1. Create virtual environment: `uv venv test_env`
2. Activate environment: `source test_env/bin/activate` (Linux/Mac)
3. Install from local build: `uv pip install --extra-index-url https://download.pytorch.org/whl/cpu dist/*.whl`
4. Test full functionality including model download
5. Deactivate and clean up test environment

## Step 9: Upload to PyPI
1. Ensure PyPI account is set up and verified
2. Upload distributions: `twine upload dist/*`
3. Verify upload on PyPI website
4. Test installation from PyPI: `uv pip install indic-asr`

## Step 10: Update Hugging Face Model Card
1. Add PyPI installation instructions to HF model card
2. Include links between HF repo and PyPI package
3. Update usage examples to show both installation methods
4. Add badges and links

## Step 11: Create GitHub Repository (Optional but Recommended)
1. Create GitHub repository for source code
2. Push package source code
3. Set up GitHub Actions for:
   - Automated testing
   - PyPI publishing on releases
   - Documentation building
4. Add GitHub releases for version management

## Step 12: Documentation and Community
1. Create comprehensive documentation website (Read the Docs, GitHub Pages)
2. Set up issue templates on GitHub
3. Create discussion forums or Discord community
4. Add contribution guidelines
5. Set up CI/CD pipelines

## Step 13: Announce and Monitor
1. Announce release on relevant platforms:
   - Hugging Face community
   - Reddit (r/MachineLearning, r/LanguageTechnology)
   - Twitter/LinkedIn
   - ASR-specific forums
2. Monitor for issues and feedback
3. Set up analytics for download tracking
4. Plan for regular updates and maintenance

## Step 14: Maintenance and Updates
1. Monitor PyPI download statistics
2. Track GitHub issues and PRs
3. Plan regular version releases
4. Update dependencies as needed
5. Maintain compatibility across Python versions
6. Handle security updates promptly

## Troubleshooting Common Issues
- Dependency conflicts: Test in multiple environments
- Model download failures: Implement retry logic
- Platform compatibility: Test on Windows, macOS, Linux
- Large model sizes: Optimize download and caching
- API changes: Maintain backward compatibility

## Security Considerations
- Code review before publishing
- Dependency vulnerability scanning
- Secure credential handling for HF uploads
- Regular security audits
- Responsible disclosure policy

## Performance Optimization
- Minimize package size
- Optimize model loading times
- Implement efficient caching
- Profile and optimize inference speed
- Consider platform-specific optimizations