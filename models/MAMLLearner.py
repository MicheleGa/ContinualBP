import torch
import torch.nn as nn


class MAMLLearner(nn.Module):
    """
    Learner wrapper for MAML-style functional forward using torch.func.functional_call.
    Supports both the BIOT encoder and a flexible prediction prediction_head (BPRegressor or BPWaveformDecoder).
    """
    def __init__(self, encoder, prediction_head):
        super().__init__()
        self.encoder = encoder
        self.prediction_head = prediction_head  # can be BPRegressor or BPWaveformDecoder

        # Capture parameter names in deterministic order
        self.encoder_named_params = list(self.encoder.named_parameters())
        self.prediction_head_named_params = list(self.prediction_head.named_parameters())

        # Flattened list of parameters (order must match .parameters())
        self._flattened_params = [p for _, p in self.encoder_named_params if p.is_floating_point() or p.is_complex()]
        self._flattened_params += [p for _, p in self.prediction_head_named_params if p.is_floating_point() or p.is_complex()]
        
        # Save names for reconstruction of param dicts
        self._encoder_names_in_order = [n for n, _ in self.encoder_named_params]
        self._prediction_head_names_in_order = [n for n, _ in self.prediction_head_named_params]

        # Guardrail: expected embedding size from encoder
        self.expected_feat_dim = getattr(self.encoder, "embed_dim", None)

    def _split_vars_to_dicts(self, vars_list):
        """
        Split a flat list of tensors into two param dicts matching encoder and prediction_head.
        """
        n_encoder = len(self._encoder_names_in_order)
        encoder_vars = vars_list[:n_encoder]
        prediction_head_vars = vars_list[n_encoder:]

        encoder_param_dict = {name: tensor for name, tensor in zip(self._encoder_names_in_order, encoder_vars)}
        prediction_head_param_dict = {name: tensor for name, tensor in zip(self._prediction_head_names_in_order, prediction_head_vars)}
        return encoder_param_dict, prediction_head_param_dict

    def forward(self, x, vars=None):
        """
        If vars is None -> standard forward.
        If vars is not None -> functional forward with provided fast weights.
        """
        if vars is None:
            feats = self.encoder(x)

            # Expect [B, T, D]
            assert feats.dim() == 3, f"prediction_head expects [B, T, D], got {list(feats.shape)}"
            
            return self.prediction_head(feats)

        # ---- Functional forward (MAML inner loop) ----
        encoder_param_dict, prediction_head_param_dict = self._split_vars_to_dicts(vars)

        feats = torch.func.functional_call(self.encoder, encoder_param_dict, (x,))

        # Expect [B, T, D]
        assert feats.dim() == 3, f"prediction_head expects [B, T, D], got {list(feats.shape)}"

        return torch.func.functional_call(self.prediction_head, prediction_head_param_dict, (feats,))