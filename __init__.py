"""Standalone MOSS BBox V3 custom-node package."""

from . import official_dsh
from .moss_bbox_v3 import YebanMossBBoxStandaloneV3
from .official_dsh import YebanOfficialDshConfig
from .provider_manager import YebanCodexProviderManager

NODE_CLASS_MAPPINGS = {
    "YebanMossBBoxShotControlV1": YebanMossBBoxStandaloneV3,
    "YebanMossBBoxProviderManager": YebanCodexProviderManager,
    "YebanMossBBoxDshConfig": YebanOfficialDshConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YebanMossBBoxShotControlV1": "Yeban MOSS BBox 景别控制 V1",
    "YebanMossBBoxProviderManager": "夜半 MOSS BBox 供应商管理器",
    "YebanMossBBoxDshConfig": "夜半 MOSS BBox 官方 DSH 配置",
}

WEB_DIRECTORY = "./web"

official_dsh.register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
