from models.vit import ViT
from models.swin_transformer import SwinTransformer


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
    return model
