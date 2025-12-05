# Indic ASR

A quantized automatic speech recognition (ASR) system for multiple Indic languages using the IndicConformer architecture.

## Installation

### CPU-only Installation (Recommended for limited resources)
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu indic-asr
```

### GPU Installation
```bash
pip install indic-asr
```

For uv users:
```bash
uv pip install --extra-index-url https://download.pytorch.org/whl/cpu indic-asr
```

## Quick Start

```python
from indic_asr import IndicConformerTranscriber

# Initialize (downloads model automatically)
transcriber = IndicConformerTranscriber()

# Transcribe audio
text = transcriber.transcribe_ctc("audio.wav", "hi")  # Hindi
print(text)
```

## Supported Languages

- Hindi (hi)
- Bengali (bn)
- Telugu (te)
- Marathi (mr)
- Tamil (ta)
- Gujarati (gu)
- Kannada (kn)
- Malayalam (ml)
- Odia (or)
- Punjabi (pa)
- Assamese (as)

## Features

- **Quantized Models**: INT8 quantization for efficient CPU inference
- **Multiple Languages**: Support for 11 Indic languages
- **Two Modes**: CTC and RNN-T decoding
- **Auto Download**: Models download automatically on first use
- **ONNX Runtime**: Optimized inference with ONNX

## Usage

### CTC Mode (Faster)
```python
text = transcriber.transcribe_ctc("audio.wav", "hi")
```

### RNN-T Mode (More Accurate)
```python
text = transcriber.transcribe_rnnt("audio.wav", "hi")
```

## Audio Requirements

- Format: WAV, MP3, FLAC, etc.
- Sample Rate: Auto-resampled to 16kHz
- Channels: Mono (auto-converted)

## Performance

- **CPU Inference**: ~50-100x real-time on modern CPUs
- **Memory**: ~200-500MB per inference
- **Model Size**: ~500MB (downloaded on first use)

## License

MIT License

## Citation

If you use this in your research, please cite:

```
@misc{indic-asr-2025,
  title={Indic ASR: Quantized Conformer for Indic Languages},
  author={Verma, Atharva},
  year={2025}
}
```