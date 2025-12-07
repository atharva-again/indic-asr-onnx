# Indic ASR Quantized

A helper package to use Quantized Indic ASR (Automatic Speech Recognition) for multiple Indic languages.

The original model was developed by AI4Bharat and can be found [here](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)

## Benchmarks

These benchmarks were conducted on Google Colab free tier with Tesla T4 GPU for Hindi.
You can use the notebooks in `scripts` directory to reproduce the results or compute for other languages. 

| *Decoding Method* | FP 32 WER | int8 WER | FP32 CER | int8 CER |
| ----------------- | --------- | -------- | -------- | -------- |
| CTC               | 0.1645    | 0.2985   | 0.0661   | 0.1698   |
| RNNT              | 0.1508    | 0.2939   | 0.0642   | 0.149    |



## Installation

### GPU Installation
```bash
pip install uv
uv pip install indic-asr-onnx 

# Specify --extra-index-url if needed
```

### CPU-only Installation (Recommended for limited resources)
```bash
pip install uv
uv pip install indic-asr-onnx --extra-index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

```python
from indic_asr_onnx import IndicTranscriber

# Initialize (downloads model automatically)
transcriber = IndicTranscriber()

# Transcribe audio using CTC head
text = transcriber.transcribe_ctc("audio.wav", "hi")  # Hindi
print(text)

# Transcribe audio using RNN-T head
text = transcriber.transcribe_rnnt("audio.wav", "hi")  # Hindi
print(text)
```

## Supported Languages

- Assamese (as)
- Bengali (bn)
- Bodo (brx)
- Dogri (doi)
- Gujarati (gu)
- Hindi (hi)
- Kannada (kn)
- Kashmiri (ks)
- Konkani (kok)
- Maithili (mai)
- Malayalam (ml)
- Manipuri (mni)
- Marathi (mr)
- Nepali (ne)
- Odia (or)
- Punjabi (pa)
- Sanskrit (sa)
- Santali (sat)
- Sindhi (sd)
- Tamil (ta)
- Telugu (te)
- Urdu (ur)

## Features

- **Quantized Models**: INT8 quantization for efficient CPU inference
- **Multiple Languages**: Support for 22 Indic languages
- **Two Modes**: CTC and RNN-T decoding
- **Auto Download**: Models download automatically on first use
- **ONNX Runtime**: Optimized inference with ONNX

## Audio Requirements

- Format: WAV, MP3, FLAC, etc.
- Sample Rate: Auto-resampled to 16kHz
- Channels: Mono (auto-converted)

## Repository Structure

### src/indic_asr_onnx.py

Code for the IndicTranscriber class.

### scripts

Collection of notebooks that I used throughout this project. 

1. `calibration-dataset.ipynb`: Script to generate calibration dataset for static PTQ.
2. `quantize-indic-conformer.ipynb`: Script to quantize IndicConformer model.
3. `evals-on-quantized-indic-conformer-asr.ipynb`: Script to evaluate quantized model.
4. `using-indic-asr-quantized.ipynb`: Script to demonstrate usage of quantized model.
