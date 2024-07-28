from timm.data.auto_augment import RandAugment


_CUSTOM_RAND_TFS = [
    'AutoContrast', 
    # 'Equalize', # del
    # 'Invert', # del
    'Rotate', 
    # 'Posterize', # del
    # 'Solarize', # del
    # 'SolarizeAdd', # del
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX', 
    'ShearY', 
    'TranslateXRel',
    'TranslateYRel',
    # 'Cutout'  # NOTE I've implement this as random erasing separately
]
