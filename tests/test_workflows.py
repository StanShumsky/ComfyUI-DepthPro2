"""The example workflows must stay loadable: links and widget counts match the node schemas."""

import json
import sys
from pathlib import Path

import pytest

from tests.test_nodes import _entry_node_list

ROOT = Path(__file__).resolve().parents[1]

def _widget_inputs(schema):
    """The inputs rendered as widgets (value types, not link sockets), in the order the frontend
    serializes them into widgets_values: non-optional widgets first (schema order), then optional
    widgets (schema order) — ComfyUI sorts inputs by optional (required then optional; see
    comfy_api.latest's required/optional partitioning and the frontend's getOrderedInputSpecs)."""
    from comfy_api.latest import io

    widget_types = (io.Float.Input, io.Int.Input, io.Combo.Input, io.String.Input, io.Boolean.Input)
    widgets = [
        inp
        for inp in schema.inputs
        if type(inp) in widget_types and (not inp.optional or inp.default is not None)
    ]
    non_optional = [i for i in widgets if not i.optional]
    optional = [i for i in widgets if i.optional]
    return non_optional + optional

@pytest.mark.parametrize("workflow", ["i2depth", "v2depth"])
def test_example_workflow(workflow):
    node_classes = {cls.GET_SCHEMA().node_id: cls for cls in _entry_node_list()}
    data = json.loads((ROOT / "example_workflows" / f"{workflow}.json").read_text())
    by_id = {n["id"]: n for n in data["nodes"]}
    links = {link[0]: link for link in data["links"]}
    for node in data["nodes"]:
        for slot, inp in enumerate(node.get("inputs") or []):
            if inp.get("link") is None:
                continue
            link = links[inp["link"]]
            assert link[3] == node["id"] and link[4] == slot
            source = by_id[link[1]]["outputs"][link[2]]
            assert source["type"] == link[5] == inp["type"]
            assert inp["link"] in (source["links"] or [])
        for slot, out in enumerate(node.get("outputs") or []):
            for link_id in out.get("links") or []:
                link = links[link_id]
                assert link[1] == node["id"] and link[2] == slot
        cls = node_classes.get(node["type"])
        if cls is None:
            continue
        widgets = _widget_inputs(cls.GET_SCHEMA())
        values = node["widgets_values"]
        assert len(values) == len(widgets), node["type"]
        for inp, value in zip(widgets, values):
            # Only combo inputs carry options (the value types do not, in the real API either).
            options = getattr(inp, "options", None)
            if isinstance(options, list):
                assert value in options, (node["type"], inp.id, value)