"""
Speaker-Conditioned Voxtral Model.

Wraps VoxtralForConditionalGeneration with:
  1. A (optionally trainable) SiAMResNet34 x-vector extractor
  2. FiLM layers that condition the Voxtral audio encoder on x-vectors

The decoder (LLM) is kept frozen; only the FiLM generators (+optionally
the encoder and/or SiAMResNet34) are trained.
"""

import math
import torch
import torch.nn as nn

from transformers import VoxtralForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from film import FiLMBank


class SpeakerConditionedVoxtral(nn.Module):
    """
    Voxtral model whose audio encoder is conditioned on speaker
    embeddings via FiLM layers.

    Pipeline
    --------
    raw_waveform  ──►  SiAMResNet34 (frozen or trainable) ──► x-vector z
                                                                    │
    mel features  ──►  VoxtralEncoder  ──layer_i──►  FiLM(h, z)  ◄─┘
                                         ...
                       ──► projector ──► LLM decoder (frozen) ──► text

    Parameters
    ----------
    voxtral_model : VoxtralForConditionalGeneration
        Pre-trained Voxtral model (loaded externally).
    xvector_model_path : str
        Path to the TorchScript SiAMResNet34 .jit file.
    film_config : dict
        Configuration for the FiLM bank:
          - hidden_dim (int): FiLM MLP hidden size (default 512)
          - mode (str): "per_layer" or "shared" (default "per_layer")
          - use_gate (bool): learnable gating (default True)
    train_xvector : bool
        Whether to allow gradients through SiAMResNet34 (default True).
    train_encoder : bool
        Whether to also train the Voxtral audio encoder (default False).
    train_projector : bool
        Whether to also train the multi-modal projector (default True).
    """

    # Normative speech types — no pathological conditioning needed
    NORMATIVE_TYPES = {"HC", "Unknown"}

    def __init__(
        self,
        voxtral_model: VoxtralForConditionalGeneration,
        xvector_model_path: str,
        film_config: dict = None,
        train_xvector: bool = True,
        train_encoder: bool = False,
        train_projector: bool = True,
    ):
        super().__init__()

        film_config = film_config or {}

        # ---- Store the full Voxtral model ----
        self.voxtral = voxtral_model
        self.config = voxtral_model.config

        # Encoder info
        enc_cfg = self.config.audio_config
        self.d_model = enc_cfg.d_model            # 1280
        self.num_layers = enc_cfg.encoder_layers   # 32

        # ---- Load SiAMResNet34 x-vector extractor ----
        # Load JIT model and move to the same device as the Voxtral model.
        # The JIT model includes internal fbank buffers that must be on the
        # same device as the input waveforms.
        xvec_device = next(voxtral_model.parameters()).device
        self.xvector_model = torch.jit.load(xvector_model_path, map_location=xvec_device)
        self.xvector_dim = 256  # SiAMResNet34 output dim
        self.train_xvector = train_xvector

        # Ensure internal buffers in the JIT x-vector model are contiguous.
        # Some TorchScript modules (e.g. fbank buffers) keep non-contiguous
        # views which cause safetensors.save_file to raise when Trainer saves
        # checkpoints. Make a conservative pass to pack any tensor-like
        # attributes under common names into contiguous tensors.
        try:
            if hasattr(self.xvector_model, "fbank"):
                fbank = getattr(self.xvector_model, "fbank")
                for name in dir(fbank):
                    # skip private/dunder attributes quickly
                    if name.startswith("__"):
                        continue
                    try:
                        val = getattr(fbank, name)
                    except Exception:
                        continue
                    if isinstance(val, torch.Tensor):
                        if not val.is_contiguous():
                            try:
                                setattr(fbank, name, val.contiguous())
                            except Exception:
                                # best-effort: ignore if we can't reassign
                                pass
        except Exception:
            # Don't crash construction if the heuristic fails; it's only a
            # best-effort safeguard for saving with safetensors.
            pass

        if not train_xvector:
            self.xvector_model.eval()
            for p in self.xvector_model.parameters():
                p.requires_grad_(False)

        # ---- FiLM bank ----
        self.film_bank = FiLMBank(
            num_layers=self.num_layers,
            spk_dim=self.xvector_dim,
            hidden_dim=film_config.get("hidden_dim", 512),
            out_dim=self.d_model,
            use_gate=film_config.get("use_gate", True),
            mode=film_config.get("mode", "per_layer"),
        )

        # Move FiLM bank to the same device/dtype as the encoder
        enc_device = next(voxtral_model.audio_tower.parameters()).device
        enc_dtype = next(voxtral_model.audio_tower.parameters()).dtype
        self.film_bank = self.film_bank.to(device=enc_device, dtype=enc_dtype)

        # ---- Expose device map so HF Trainer skips DataParallel wrapping ----
        # When Voxtral is loaded with device_map="auto", it gets an
        # hf_device_map attribute.  The Trainer checks for this and
        # sets self.is_model_parallelized = True, which prevents it
        # from wrapping the model in nn.DataParallel.
        if hasattr(voxtral_model, "hf_device_map"):
            self.hf_device_map = voxtral_model.hf_device_map

        # ---- Freeze / unfreeze components ----
        self._configure_trainable(train_encoder, train_projector)

    def _configure_trainable(self, train_encoder: bool, train_projector: bool):
        """Freeze decoder; optionally freeze encoder and projector."""
        # Always freeze the LLM decoder
        for p in self.voxtral.language_model.parameters():
            p.requires_grad_(False)

        # Encoder
        encoder = self.voxtral.audio_tower
        for p in encoder.parameters():
            p.requires_grad_(train_encoder)

        # Projector
        projector = self.voxtral.multi_modal_projector
        for p in projector.parameters():
            p.requires_grad_(train_projector)

        # FiLM bank is always trainable (this is the whole point)
        for p in self.film_bank.parameters():
            p.requires_grad_(True)

        # Print summary
        self._print_param_summary()

    def _print_param_summary(self):
        """Print trainable / frozen parameter counts."""
        components = {
            "SiAMResNet34 (x-vector)": self.xvector_model.parameters(),
            "Voxtral Encoder": self.voxtral.audio_tower.parameters(),
            "Multi-Modal Projector": self.voxtral.multi_modal_projector.parameters(),
            "FiLM Bank": self.film_bank.parameters(),
            "LLM Decoder": self.voxtral.language_model.parameters(),
        }

        print("\n" + "=" * 70)
        print("SPEAKER-CONDITIONED VOXTRAL — PARAMETER SUMMARY")
        print("=" * 70)

        total_trainable = 0
        total_all = 0
        for name, params in components.items():
            params = list(params)
            n_all = sum(p.numel() for p in params)
            n_train = sum(p.numel() for p in params if p.requires_grad)
            total_all += n_all
            total_trainable += n_train
            status = "✓ trainable" if n_train > 0 else "✗ frozen"
            print(f"  {name:35s}  {n_train:>12,} / {n_all:>12,}  {status}")

        print("-" * 70)
        pct = total_trainable / total_all * 100 if total_all > 0 else 0
        print(f"  {'TOTAL':35s}  {total_trainable:>12,} / {total_all:>12,}  ({pct:.2f}%)")
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Core forward: run encoder with FiLM conditioning
    # ------------------------------------------------------------------
    def _encode_with_film(self, input_features: torch.Tensor,
                          xvector: torch.Tensor,
                          chunks_per_sample: torch.Tensor = None) -> torch.Tensor:
        """
        Run the Voxtral audio encoder with FiLM conditioning
        injected after every transformer layer.

        Parameters
        ----------
        input_features : (N_chunks, n_mels, T_mel)
            Mel spectrogram features.  N_chunks >= B because long audio
            is split into 30-second chunks by the Voxtral processor.
        xvector : (B, 256)
            Speaker embeddings — one per *sample* (not per chunk).
        chunks_per_sample : (B,) int tensor or None
            Number of chunks per sample.  Used to expand x-vectors to
            match N_chunks via repeat_interleave.  If None (all samples
            have 1 chunk), xvector is used as-is.

        Returns
        -------
        hidden_states : (B, T_enc, d_model)
        """
        encoder = self.voxtral.audio_tower

        # --- Conv stem ---
        input_features = input_features.to(
            dtype=encoder.conv1.weight.dtype,
            device=encoder.conv1.weight.device,
        )
        inputs_embeds = nn.functional.gelu(encoder.conv1(input_features))
        inputs_embeds = nn.functional.gelu(encoder.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # (B, T, D)

        # --- Positional embeddings ---
        embed_pos = encoder.embed_positions.weight
        hidden_states = (inputs_embeds + embed_pos).to(inputs_embeds.dtype)
        hidden_states = nn.functional.dropout(
            hidden_states, p=encoder.dropout, training=encoder.training
        )

        # Cast x-vector to same dtype
        z = xvector.to(dtype=hidden_states.dtype)

        # --- Expand x-vectors to match chunks ---
        # input_features may have more rows than batch size when audio > 30s
        # is split into multiple chunks.  Repeat each sample's x-vector for
        # all of its chunks so that z.shape[0] == hidden_states.shape[0].
        if chunks_per_sample is not None and z.shape[0] != hidden_states.shape[0]:
            z = torch.repeat_interleave(z, chunks_per_sample.to(z.device), dim=0)

        # --- Transformer layers + FiLM ---
        # Note: with device_map="auto", encoder layers may be on different
        # GPUs.  We move the x-vector (and FiLM computation) to follow
        # wherever hidden_states land after each layer.
        for idx, encoder_layer in enumerate(encoder.layers):
            # The layer might live on a different device (device_map="auto").
            # Use a try/except as a safety net in case the layer has no
            # iterable parameters (e.g. inside a DataParallel replica).
            try:
                layer_device = next(encoder_layer.parameters()).device
            except StopIteration:
                layer_device = hidden_states.device
            hidden_states = hidden_states.to(layer_device)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=None,
                layer_head_mask=None,
            )
            hidden_states = layer_outputs[0]

            # Apply FiLM conditioning after each layer.
            # Move z to the same device as hidden_states for this layer.
            z_local = z.to(hidden_states.device)
            hidden_states = self.film_bank.apply(idx, hidden_states, z_local)

        hidden_states = encoder.layer_norm(hidden_states.to(encoder.layer_norm.weight.device))

        return hidden_states

    # Maximum waveform length for x-vector extraction (15 s @ 16 kHz).
    # Speaker embeddings are robust with ≥5 s of audio; truncating longer
    # utterances avoids GPU OOM from outliers (e.g., 180 s recordings).
    XVEC_MAX_SAMPLES = 15 * 16_000  # 240 000 samples

    def _extract_xvectors(self, raw_waveforms: torch.Tensor,
                          is_normative: torch.Tensor) -> torch.Tensor:
        """
        Extract x-vectors from raw waveforms.

        For normative speakers, we use a zero vector so that the FiLM gate
        learns to produce identity modulation for them.

        Parameters
        ----------
        raw_waveforms : (B, T_samples)
            Raw audio waveforms at 16 kHz.
        is_normative : (B,) bool tensor
            True for normative (HC/Unknown) speakers.

        Returns
        -------
        xvectors : (B, 256)
        """
        # Ensure waveforms are float32 — SiAMResNet34 fbank expects float32
        waveforms_f32 = raw_waveforms.float()

        # Truncate to max length to prevent OOM on very long utterances.
        # X-vectors only need ~10-15 s of audio for a reliable embedding.
        if waveforms_f32.shape[-1] > self.XVEC_MAX_SAMPLES:
            waveforms_f32 = waveforms_f32[..., :self.XVEC_MAX_SAMPLES]

        if self.train_xvector:
            xvectors = self.xvector_model(waveforms_f32)  # (B, 256)
        else:
            with torch.no_grad():
                xvectors = self.xvector_model(waveforms_f32)  # (B, 256)

        # For normative speakers, zero out the x-vector.
        # The FiLM generators (initialized near identity + gate near 0)
        # will learn that zero input → no modulation.
        if is_normative is not None and is_normative.any():
            mask = is_normative.unsqueeze(-1).to(xvectors.dtype)  # (B, 1)
            xvectors = xvectors * (1.0 - mask)

        return xvectors

    # ------------------------------------------------------------------
    # Full forward pass (mirrors VoxtralForConditionalGeneration.forward)
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        raw_waveforms=None,
        is_normative=None,
        labels=None,
        **kwargs,
    ):
        """
        Forward pass with speaker-conditioned encoding.

        Extra inputs compared to vanilla Voxtral:
            raw_waveforms : (B, T_samples) — raw audio for x-vector extraction
            is_normative  : (B,) bool — whether each sample is normative speech
        """
        # 1) Get x-vectors from raw waveforms
        if raw_waveforms is not None:
            xvectors = self._extract_xvectors(raw_waveforms, is_normative)
        else:
            raise ValueError(
                "raw_waveforms must be provided for speaker-conditioned Voxtral. "
                "The data collator should include raw audio waveforms."
            )

        # 2) Run encoder with FiLM conditioning
        if input_features is not None and input_ids is not None:
            # Voxtral chunks long audio (>30 s) into multiple 30-s segments.
            # input_features.shape[0] = total chunks (>= B).
            # We need to know how many chunks belong to each sample so we
            # can repeat the per-sample x-vector accordingly.
            n_chunks = input_features.shape[0]
            B = raw_waveforms.shape[0]
            chunks_per_sample = None
            if n_chunks != B:
                # Count audio tokens per sample; each chunk → tokens_per_chunk audio tokens
                audio_token_id = self.config.audio_token_id
                audio_counts = (input_ids == audio_token_id).sum(dim=1)  # (B,)
                tokens_per_chunk = audio_counts.sum().item() // n_chunks
                chunks_per_sample = audio_counts // tokens_per_chunk  # (B,)

            audio_hidden = self._encode_with_film(
                input_features, xvectors, chunks_per_sample
            )

            # Reshape + project  (same as VoxtralForConditionalGeneration.get_audio_features)
            audio_hidden = audio_hidden.reshape(
                -1, self.config.audio_config.intermediate_size
            )
            audio_embeds = self.voxtral.multi_modal_projector(audio_hidden)

            # Get text embeddings
            inputs_embeds = self.voxtral.get_input_embeddings()(input_ids)

            # Replace audio token placeholders with conditioned audio embeddings
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device),
                audio_embeds.to(inputs_embeds.device),
            )
        else:
            inputs_embeds = self.voxtral.get_input_embeddings()(input_ids)

        # 3) Run LLM decoder
        outputs = self.voxtral.language_model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )

        return outputs

    # ------------------------------------------------------------------
    # Generation (for inference / evaluation)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        raw_waveforms=None,
        is_normative=None,
        **generate_kwargs,
    ):
        """
        Generate text with speaker-conditioned encoding.
        Replaces the audio encoder output with FiLM-conditioned features,
        then delegates to the LLM's generate().
        """
        # 1) x-vectors
        if raw_waveforms is not None:
            xvectors = self._extract_xvectors(raw_waveforms, is_normative)
        else:
            raise ValueError("raw_waveforms required for generation.")

        # 2) FiLM-conditioned encoder
        if input_features is not None and input_ids is not None:
            # Handle audio chunking (same logic as forward)
            n_chunks = input_features.shape[0]
            B = raw_waveforms.shape[0]
            chunks_per_sample = None
            if n_chunks != B:
                audio_token_id = self.config.audio_token_id
                audio_counts = (input_ids == audio_token_id).sum(dim=1)
                tokens_per_chunk = audio_counts.sum().item() // n_chunks
                chunks_per_sample = audio_counts // tokens_per_chunk

            audio_hidden = self._encode_with_film(
                input_features, xvectors, chunks_per_sample
            )
            audio_hidden = audio_hidden.reshape(
                -1, self.config.audio_config.intermediate_size
            )
            audio_embeds = self.voxtral.multi_modal_projector(audio_hidden)

            inputs_embeds = self.voxtral.get_input_embeddings()(input_ids)
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device),
                audio_embeds.to(inputs_embeds.device),
            )
        else:
            inputs_embeds = self.voxtral.get_input_embeddings()(input_ids)

        # 3) Generate via the LLM
        return self.voxtral.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing on the Voxtral encoder."""
        self.voxtral.gradient_checkpointing_enable(**kwargs)

    @property
    def device(self):
        return next(self.voxtral.parameters()).device

    @property
    def dtype(self):
        return next(self.voxtral.parameters()).dtype
