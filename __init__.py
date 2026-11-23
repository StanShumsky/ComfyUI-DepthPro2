"""ComfyUI nodes for Apple Depth Pro 2 metric monocular depth.

The whole pack registers through the V3 entrypoint below. ComfyUI's loader registers a custom
node module as either V1 (NODE_CLASS_MAPPINGS) or V3 (comfy_entrypoint) - it checks V1 first
and returns, so a module with both would silently lose its V3 nodes; this pack therefore has no
NODE_CLASS_MAPPINGS of its own and registers everything here: the nine io.ComfyNode classes in
nodes.py.
"""
from comfy_api.latest import ComfyExtension

from . import nodes

__all__ = ["comfy_entrypoint"]


class DepthPro2Pack(ComfyExtension):
    async def get_node_list(self):
        return [
            nodes.DepthPro2Loader,
            nodes.DepthPro2Estimate,
            nodes.DepthPro2Video,
            nodes.DepthPro2FocalLength,
            nodes.DepthPro2Colorize,
            nodes.DepthPro2PositionPass,
            nodes.DepthPro2Save16,
            nodes.DepthPro2SaveFloat,
            nodes.DepthPro2SavePLY,
        ]


async def comfy_entrypoint() -> DepthPro2Pack:
    return DepthPro2Pack()
