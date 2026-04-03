#!/usr/bin/env python3

"""
gait_classifier.py — Real-time gait classification using a sliding window TCN.

Pure inference class — no ROS2, no robot interface, no side effects.
Instantiate once, call predict(features) every frame.
"""

import os
import sys
import warnings
from collections import deque

import numpy as np
import pandas as pd
import torch

# Directory that contains this file — used to locate ModelV1.pth and train.py
# regardless of working directory or install location.
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


class GaitClassifier:
    """
    Sliding-window gait classifier backed by a Temporal Convolutional Network.

    Parameters
    ----------
    model_path      : path to the .pth checkpoint saved during training
    sequence_length : number of frames in the sliding window (default 60)
    device          : 'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(
        self,
        model_path:      str,
        sequence_length: int = 60,
        device:          str | None = None,
    ):
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Resolve model_path: if the provided value is not an existing file
        # (e.g. a directory, a stale relative path, or just a filename), fall
        # back to ModelV1.pth sitting next to this module.
        if not os.path.isfile(model_path):
            model_path = os.path.join(_MODEL_DIR, 'ModelV1.pth')

        # Ensure train.py (co-located with this file) is importable before
        # torch.load — pickle resolves module names during deserialisation.
        if _MODEL_DIR not in sys.path:
            sys.path.insert(0, _MODEL_DIR)

        # The checkpoint was pickled when the training script was named
        # 'training.py'. Register it under both names so torch.load can
        # deserialise classes that were saved with the old module path.
        import train as _train_mod
        sys.modules.setdefault('training', _train_mod)
        sys.modules.setdefault('training.train', _train_mod)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False)

        self.preprocessor    = checkpoint['preprocessor']
        self.sequence_length = sequence_length

        from train import GaitTCN

        num_features = len(self.preprocessor.processed_feature_names)
        num_classes  = len(self.preprocessor.label_encoder.classes_)

        self.model = GaitTCN(
            num_features=num_features,
            num_classes=num_classes,
            num_channels=[64, 128, 256],
            kernel_size=7,
            dropout=0.3,
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.frame_buffer: deque = deque(maxlen=sequence_length)
        self.is_ready: bool = False

    # ── Feature preparation ───────────────────────────────────────────────────

    def _prepare_features(self, raw_features: dict) -> np.ndarray:
        """
        Map raw feature dict → scaled numpy array expected by the model.
        Handles the support_type categorical encoding internally.
        """
        feature_dict: dict = {}

        for orig_name in self.preprocessor.original_feature_names:
            if orig_name == 'support_type':
                val = raw_features.get('support_type', 'double')
                if val not in self.preprocessor.support_encoder.classes_:
                    val = 'double'
                feature_dict['support_type_encoded'] = (
                    self.preprocessor.support_encoder.transform([val])[0])
            else:
                processed_name = orig_name
                feature_dict[processed_name] = float(
                    raw_features.get(orig_name, 0.0))

        feature_df    = pd.DataFrame([feature_dict])[
            self.preprocessor.processed_feature_names]
        feature_array = self.preprocessor.scaler.transform(feature_df)
        return feature_array.flatten()

    # ── Buffer fill (call at data rate) ───────────────────────────────────────

    def add_frame(self, raw_features: dict) -> None:
        """
        Append one frame to the sliding window buffer.

        Must be called at the raw data rate (e.g. 240 Hz / mocap rate) so the
        buffer always contains the most recent N consecutive frames as the model
        expects.  Inference is decoupled — call infer() separately.
        """
        self.frame_buffer.append(self._prepare_features(raw_features))
        if len(self.frame_buffer) == self.sequence_length:
            self.is_ready = True

    # ── Inference (call at classifier rate) ───────────────────────────────────

    def infer(self) -> tuple[str | None, float, dict]:
        """
        Run the TCN forward pass on the current buffer contents.

        Returns
        -------
        prediction  : gait label string, or None if buffer not yet full
        confidence  : softmax confidence of the top class [0, 1]
        info        : per-class probabilities dict, or {'buffering': N} if not ready
        """
        if not self.is_ready:
            return None, 0.0, {'buffering': self.sequence_length - len(self.frame_buffer)}

        sequence        = np.array(list(self.frame_buffer))
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs       = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        predicted_idx = predicted_idx.item()
        confidence    = confidence.item()
        prediction    = self.preprocessor.label_encoder.inverse_transform(
            [predicted_idx])[0]

        prob_dict = {
            name: probabilities[0, idx].item()
            for idx, name in enumerate(self.preprocessor.label_encoder.classes_)
        }

        return prediction, confidence, prob_dict

    def predict(
        self,
        raw_features: dict,
    ) -> tuple[str | None, float, dict]:
        """Add one frame and immediately infer. Convenience wrapper."""
        self.add_frame(raw_features)
        return self.infer()

    def reset(self) -> None:
        """Clear the frame buffer — call this on tracking loss recovery."""
        self.frame_buffer.clear()
        self.is_ready = False