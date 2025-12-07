import json
import os

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from huggingface_hub import snapshot_download


class IndicTranscriber:
    def __init__(
        self, model_dir=None, repo_id="atharva-again/indic-conformer-600m-quantized"
    ):
        """
        Initialize the transcriber.

        Args:
            model_dir: Local path to model directory. If None, downloads from HF.
            repo_id: HF repo ID if downloading.
        """
        if model_dir is None:
            print(f"Downloading model from {repo_id}...")
            model_dir = snapshot_download(repo_id=repo_id)

        self.model_dir = model_dir

        # Device and Provider Detection
        if (
            torch.cuda.is_available()
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            self.device = torch.device("cuda")
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.providers = ["CPUExecutionProvider"]
            print("Using CPU")

        # Preprocessor
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=0.0,
            f_max=8000.0,
            n_mels=80,
            window_fn=torch.hann_window,
            power=2.0,
        ).to(self.device)

        # Lazy load ONNX sessions
        self.encoder_sess = None
        self.ctc_sess = None
        self.rnnt_sess = None
        self.joint_enc_sess = None
        self.joint_pred_sess = None
        self.joint_pre_net_sess = None
        self.joint_post_net_sess = None

    def _load_ctc_models(self):
        if self.encoder_sess is None:
            enc_path = os.path.join(
                self.model_dir, "onnx", "encoder_quantized_int8.onnx"
            )
            self.encoder_sess = ort.InferenceSession(enc_path, providers=self.providers)
        if self.ctc_sess is None:
            ctc_path = os.path.join(
                self.model_dir, "onnx", "ctc_decoder_quantized_int8.onnx"
            )
            self.ctc_sess = ort.InferenceSession(ctc_path, providers=self.providers)

    def _load_rnnt_models(self, lang):
        self._load_ctc_models()  # Encoder is shared
        if self.rnnt_sess is None:
            rnnt_path = os.path.join(
                self.model_dir, "onnx", "rnnt_decoder_quantized_int8.onnx"
            )
            self.rnnt_sess = ort.InferenceSession(rnnt_path, providers=self.providers)
        if self.joint_enc_sess is None:
            joint_enc_path = os.path.join(
                self.model_dir, "onnx", "joint_enc_quantized_int8.onnx"
            )
            self.joint_enc_sess = ort.InferenceSession(
                joint_enc_path, providers=self.providers
            )
        if self.joint_pred_sess is None:
            joint_pred_path = os.path.join(
                self.model_dir, "onnx", "joint_pred_quantized_int8.onnx"
            )
            self.joint_pred_sess = ort.InferenceSession(
                joint_pred_path, providers=self.providers
            )
        if self.joint_pre_net_sess is None:
            joint_pre_net_path = os.path.join(
                self.model_dir, "onnx", "joint_pre_net_quantized_int8.onnx"
            )
            self.joint_pre_net_sess = ort.InferenceSession(
                joint_pre_net_path, providers=self.providers
            )
        if self.joint_post_net_sess is None:
            joint_post_net_path = os.path.join(
                self.model_dir,
                "onnx",
                f"adapters/joint_post_net_{lang}_quantized_int8.onnx",
            )
            self.joint_post_net_sess = ort.InferenceSession(
                joint_post_net_path, providers=self.providers
            )

    def _preprocess_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            ).to(self.device)
            waveform = resampler(waveform)

        features = self.mel_transform(waveform)
        features = torch.log(features + 1e-9)
        mean = features.mean(dim=2, keepdims=True)
        stddev = features.std(dim=2, keepdim=True) + 1e-5
        features = (features - mean) / stddev
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def transcribe_ctc(self, audio_path, lang):
        # Load vocab and masks for the language
        vocab_path = os.path.join(self.model_dir, "config", "vocab.json")
        masks_path = os.path.join(self.model_dir, "config", "language_masks.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)[lang]
        with open(masks_path, "r", encoding="utf-8") as f:
            masks = json.load(f)[lang]

        self._load_ctc_models()
        features = self._preprocess_audio(audio_path)
        length = np.array([features.shape[1]], dtype=np.int64)
        features = np.expand_dims(features, axis=0)

        # Encoder
        enc_inputs = self.encoder_sess.get_inputs()
        enc_dict = {enc_inputs[0].name: features}
        if len(enc_inputs) > 1:
            enc_dict[enc_inputs[1].name] = length
        enc_out = self.encoder_sess.run(None, enc_dict)[0]

        # CTC
        ctc_inputs = self.ctc_sess.get_inputs()
        ctc_dict = {ctc_inputs[0].name: enc_out}
        if len(ctc_inputs) > 1:
            ctc_dict[ctc_inputs[1].name] = length
        logits = self.ctc_sess.run(None, ctc_dict)[0]

        # Decode
        mask = np.array(masks, dtype=bool)
        logits_sliced = logits[:, :, mask]
        pred_ids = np.argmax(logits_sliced, axis=-1)[0]

        tokens = []
        prev = None
        for idx in pred_ids:
            if idx != prev and idx != 256 and idx < len(vocab):
                tokens.append(vocab[idx])
            prev = idx
        return "".join(tokens).replace("▁", " ").strip()

    def transcribe_rnnt(self, audio_path, lang):
        # Load vocab and masks for the language
        vocab_path = os.path.join(self.model_dir, "config", "vocab.json")
        masks_path = os.path.join(self.model_dir, "config", "language_masks.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)[lang]
        with open(masks_path, "r", encoding="utf-8") as f:
            masks = json.load(f)[lang]

        self._load_rnnt_models(lang)
        features = self._preprocess_audio(audio_path)
        length = np.array([features.shape[1]], dtype=np.int64)
        features = np.expand_dims(features, axis=0)

        # Encoder
        enc_inputs = self.encoder_sess.get_inputs()
        enc_dict = {enc_inputs[0].name: features}
        if len(enc_inputs) > 1:
            enc_dict[enc_inputs[1].name] = length
        enc_out = self.encoder_sess.run(None, enc_dict)[0]
        enc_out_transposed = enc_out.transpose(0, 2, 1)

        # Joint Enc
        joint_enc_dict = {self.joint_enc_sess.get_inputs()[0].name: enc_out_transposed}
        enc_output = self.joint_enc_sess.run(None, joint_enc_dict)[0]
        T = enc_output.shape[1]

        # Greedy RNNT decoding
        BLANK_ID = 256
        predicted_tokens = [BLANK_ID]
        t = 0
        max_symbols = 100

        # Initial decoder
        decoder_input = np.array([predicted_tokens], dtype=np.int32)
        target_length = np.array([len(predicted_tokens)], dtype=np.int32)
        rnnt_dict = {
            "targets": decoder_input,
            "target_length": target_length,
            "states.1": np.zeros((2, 1, 640), dtype=np.float32),
            "onnx::Slice_3": np.zeros((2, 1, 640), dtype=np.float32),
        }
        decoder_out = self.rnnt_sess.run(None, rnnt_dict)[0].transpose(0, 2, 1)
        last_emb = decoder_out[:, -1:, :]

        joint_pred_dict = {self.joint_pred_sess.get_inputs()[0].name: last_emb}
        pred_current = self.joint_pred_sess.run(None, joint_pred_dict)[0]

        while t < T and len(predicted_tokens) < max_symbols:
            enc_current = enc_output[:, t : t + 1, :]
            joint_input = enc_current + pred_current

            joint_pre_dict = {self.joint_pre_net_sess.get_inputs()[0].name: joint_input}
            pre_out = self.joint_pre_net_sess.run(None, joint_pre_dict)[0]

            joint_post_dict = {self.joint_post_net_sess.get_inputs()[0].name: pre_out}
            logits = self.joint_post_net_sess.run(None, joint_post_dict)[0]
            k = np.argmax(logits[0, 0, :])

            if k == BLANK_ID:
                t += 1
            else:
                predicted_tokens.append(int(k))
                decoder_input = np.array([predicted_tokens], dtype=np.int32)
                target_length = np.array([len(predicted_tokens)], dtype=np.int32)
                rnnt_dict["targets"] = decoder_input
                rnnt_dict["target_length"] = target_length
                decoder_out = self.rnnt_sess.run(None, rnnt_dict)[0].transpose(0, 2, 1)
                last_emb = decoder_out[:, -1:, :]
                joint_pred_dict = {self.joint_pred_sess.get_inputs()[0].name: last_emb}
                pred_current = self.joint_pred_sess.run(None, joint_pred_dict)[0]

        tokens = []
        prev = None
        for idx in predicted_tokens[1:]:
            if idx != prev and idx != BLANK_ID and idx < len(vocab):
                tokens.append(vocab[idx])
            prev = idx
        return "".join(tokens).replace("▁", " ").strip()
