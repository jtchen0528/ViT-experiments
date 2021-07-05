from torch.functional import split
from .vit import ViT
from .swin_transformer import SwinTransformer
from .CvT import CvT
from .dino import Dino
from .msvit import MsViT


def create_model(model_name='vit', img_size=64, patch_size=8, num_classes=1000):
    splits = model_name.split('_')
    head = None
    if len(splits) != 1:
        model_name = splits[1]
        head = splits[0]
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
    elif model_name == 'vil':
        model = MsViT(
            arch = 'l1,h3,d96,n1,s1,g1,p4,f7,a0_l2,h3,d192,n2,s1,g1,p2,f7,a0_l3,h6,d384,n8,s0,g1,p2,f7,a0_l4,h12,d768,n1,s0,g0,p2,f7,a0',
            img_size=img_size,
            in_chans=3,
            num_classes=num_classes
        )
    if head != None:
        if head == 'Dino':
            model = Dino(
                model,
                image_size=img_size,
                # hidden layer name or index, from which to extract the embedding
                hidden_layer='to_latent',
                projection_hidden_size=img_size,      # projector network hidden dimension
                projection_layers=4,             # number of layers in projection network
                # output logits dimensions (referenced as K in paper)
                num_classes_K=num_classes,
                student_temp=0.9,                # student temperature
                # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
                teacher_temp=0.04,
                # upper bound for local crop - 0.4 was recommended in the paper
                local_upper_crop_scale=0.4,
                # lower bound for global crop - 0.5 was recommended in the paper
                global_lower_crop_scale=0.5,
                # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
                moving_average_decay=0.9,
                # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
                center_moving_average_decay=0.9,
            )
        
    return model
