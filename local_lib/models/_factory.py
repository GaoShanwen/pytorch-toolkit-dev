from typing import Any, Dict, Optional, Union
from timm.models._pretrained import PretrainedCfg
from timm.models import create_model

def create_owner_model(
        model_name: str,
        pretrained: bool = False,
        pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
        pretrained_cfg_overlay:  Optional[Dict[str, Any]] = None,
        checkpoint_path: str = '',
        scriptable: Optional[bool] = None,
        exportable: Optional[bool] = None,
        no_jit: Optional[bool] = None,
        **kwargs,
):
    return create_model(
        model_name, pretrained, pretrained_cfg, pretrained_cfg_overlay, 
        checkpoint_path, scriptable, exportable, no_jit, **kwargs,
    )