from .vit import ViT
from .swin_transformer import SwinTransformer
from .CvT import CvT


def create_model(model_name='vit', img_size=64, patch_size=8, num_classes=1000):
    if model_name == 'vit':
        model = ViT(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif model_name == 'swin':
        model = SwinTransformer(
            hidden_dim=96,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            channels=3,
            num_classes=num_classes,
            head_dim=64,
            window_size=patch_size,
            downscaling_factors=(2, 2, 2, 1),
            relative_pos_embedding=True
        )
    elif model_name == 'cvt':
        model = CvT(
            image_size=img_size,
            in_channels=3,
            num_classes=num_classes,
            dim=64,
            heads=[1, 3, 6],
            kernels=[7, 3, 3],
            strides=[4, 2, 2],
            depth=[1, 2, 10]
        )
    return model
