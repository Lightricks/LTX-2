from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.model.transformer.attention import AttentionFunction
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.utils import check_config_value


class LTXModelConfigurator(ModelConfigurator[LTXModel]):
    """
    Configurator for LTX model.
    Used to create an LTX model from a configuration dictionary.
    """

    @classmethod
    def from_config(cls: type[LTXModel], config: dict) -> LTXModel:
        config = config.get("transformer", {})

        check_config_value(config, "dropout", 0.0)
        check_config_value(config, "attention_bias", True)
        check_config_value(config, "num_vector_embeds", None)
        check_config_value(config, "activation_fn", "gelu-approximate")
        check_config_value(config, "num_embeds_ada_norm", 1000)
        check_config_value(config, "use_linear_projection", False)
        check_config_value(config, "only_cross_attention", False)
        check_config_value(config, "cross_attention_norm", True)
        check_config_value(config, "double_self_attention", False)
        check_config_value(config, "upcast_attention", False)
        check_config_value(config, "standardization_norm", "rms_norm")
        check_config_value(config, "norm_elementwise_affine", False)
        check_config_value(config, "qk_norm", "rms_norm")
        check_config_value(config, "positional_embedding_type", "rope")
        check_config_value(config, "use_audio_video_cross_attention", True)
        check_config_value(config, "share_ff", False)
        check_config_value(config, "av_cross_ada_norm", True)
        check_config_value(config, "use_middle_indices_grid", True)

        return LTXModel(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=config.get("num_attention_heads", 32),
            attention_head_dim=config.get("attention_head_dim", 128),
            in_channels=config.get("in_channels", 128),
            out_channels=config.get("out_channels", 128),
            num_layers=config.get("num_layers", 48),
            cross_attention_dim=config.get("cross_attention_dim", 4096),
            norm_eps=config.get("norm_eps", 1e-06),
            attention_type=AttentionFunction(config.get("attention_type", "default")),
            caption_channels=config.get("caption_channels", 3840),
            positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=config.get("positional_embedding_max_pos", [20, 2048, 2048]),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", True),
            audio_num_attention_heads=config.get("audio_num_attention_heads", 32),
            audio_attention_head_dim=config.get("audio_attention_head_dim", 64),
            audio_in_channels=config.get("audio_in_channels", 128),
            audio_out_channels=config.get("audio_out_channels", 128),
            audio_cross_attention_dim=config.get("audio_cross_attention_dim", 2048),
            audio_positional_embedding_max_pos=config.get("audio_positional_embedding_max_pos", [20]),
            av_ca_timestep_scale_multiplier=config.get("av_ca_timestep_scale_multiplier", 1),
            rope_type=LTXRopeType(config.get("rope_type", "interleaved")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
            apply_gated_attention=config.get("apply_gated_attention", False),
        )


class LTXVideoOnlyModelConfigurator(ModelConfigurator[LTXModel]):
    """
    Configurator for LTX video only model.
    Used to create an LTX video only model from a configuration dictionary.
    """

    @classmethod
    def from_config(cls: type[LTXModel], config: dict) -> LTXModel:
        config = config.get("transformer", {})

        check_config_value(config, "dropout", 0.0)
        check_config_value(config, "attention_bias", True)
        check_config_value(config, "num_vector_embeds", None)
        check_config_value(config, "activation_fn", "gelu-approximate")
        check_config_value(config, "num_embeds_ada_norm", 1000)
        check_config_value(config, "use_linear_projection", False)
        check_config_value(config, "only_cross_attention", False)
        check_config_value(config, "cross_attention_norm", True)
        check_config_value(config, "double_self_attention", False)
        check_config_value(config, "upcast_attention", False)
        check_config_value(config, "standardization_norm", "rms_norm")
        check_config_value(config, "norm_elementwise_affine", False)
        check_config_value(config, "qk_norm", "rms_norm")
        check_config_value(config, "positional_embedding_type", "rope")
        check_config_value(config, "use_middle_indices_grid", True)

        return LTXModel(
            model_type=LTXModelType.VideoOnly,
            num_attention_heads=config.get("num_attention_heads", 32),
            attention_head_dim=config.get("attention_head_dim", 128),
            in_channels=config.get("in_channels", 128),
            out_channels=config.get("out_channels", 128),
            num_layers=config.get("num_layers", 48),
            cross_attention_dim=config.get("cross_attention_dim", 4096),
            norm_eps=config.get("norm_eps", 1e-06),
            attention_type=AttentionFunction(config.get("attention_type", "default")),
            caption_channels=config.get("caption_channels", 3840),
            positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=config.get("positional_embedding_max_pos", [20, 2048, 2048]),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", True),
            rope_type=LTXRopeType(config.get("rope_type", "interleaved")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
            apply_gated_attention=config.get("apply_gated_attention", False),
        )


LTXV_MODEL_COMFY_RENAMING_MAP = (
    SDOps("LTXV_MODEL_COMFY_PREFIX_MAP")
    .with_matching(prefix="model.diffusion_model.")
    .with_replacement("model.diffusion_model.", "")
)
