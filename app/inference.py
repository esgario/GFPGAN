import torch
from attrdict import AttrDict

from gfpgan import GFPGANer

RESTORER_NET = None


def run_inference(image, enable_realesrgan):
    """Run Inference for GFPGAN."""

    global RESTORER_NET

    args = {
        "upscale": 2,
        "arch": "clean",
        "channel": 2,
        "model_path": "experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth",
        "bg_upsampler": "realesrgan",
        "bg_tile": 400,
        "only_center_face": False,
        "aligned": False,
        "paste_back": True,
    }

    args = AttrDict(args)

    # background upsampler
    if enable_realesrgan:
        print("Using RealesRGAN")
        bg_upsampler = None
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=args.bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=False)
    else:
        print("Not using RealesRGAN")
        bg_upsampler = None

    # set up GFPGAN restorer
    if RESTORER_NET is None:
        RESTORER_NET = GFPGANer(
            model_path=args.model_path,
            upscale=args.upscale,
            arch=args.arch,
            channel_multiplier=args.channel,
            bg_upsampler=bg_upsampler)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = RESTORER_NET.enhance(
        image, has_aligned=args.aligned, only_center_face=args.only_center_face, paste_back=args.paste_back)

    return restored_img, cropped_faces, restored_faces
