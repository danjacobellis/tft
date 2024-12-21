from tft.models import TFTClassifier
from pytorch_wavelets import DWTForward, DWTInverse
from tft.transforms import WPT2D, IWPT2D
from timm.models.registry import register_model

# If these parameters are static or can be inferred from your hparams, define them here
J = 4
wave = 'bior4.4'
mode = 'periodization'

@register_model
def tft2d(pretrained=False, **kwargs):
    """
    Create a TFT model that can be recognized by timm.
    We rely on keyword arguments (kwargs) passed from the timm training script to configure the model.
    """
    # Extract custom arguments you might have included in train_hparams.yaml by using --model-kwargs or a config object.
    # For example, if you add `model_kwargs: { embed_dim: 512, ... }` in the yaml or specify on command line:
    embed_dim = kwargs.pop('embed_dim', 512)
    class_num = kwargs.pop('num_classes', 1000)  # timm uses num_classes
    J_level = kwargs.pop('J', J)  # wavelet decomposition level
    wave_name = kwargs.pop('wave', wave)
    mode_name = kwargs.pop('mode', mode)

    # Instantiate your wavelet transforms
    wt = DWTForward(J=1, mode=mode_name, wave=wave_name)
    iwt = DWTInverse(mode=mode_name, wave=wave_name)

    # Move iwt to device after model creation in the training script, or do it inside the forward call as needed.
    wpt = WPT2D(wt)
    # If your model requires device specs, handle in the training script after creation

    # Create your TFTClassifier model
    config = type('', (), {})()  # A dummy config object if needed, or simply pass arguments
    config.embed_dim = embed_dim
    config.classifier_num_classes = class_num
    config.J = J_level
    config.channels = 3
    config.crop_size=256
    config.dim_head=64
    # ... set any other config parameters here or pass via kwargs

    model = TFTClassifier(config, wpt)
    return model
