"""The shipped blueprints (subgraphs/*.json) must stay loadable: wrapper mirrors the definition,
proxy links resolve, and every internal socket matches the live node schemas."""

import json
import sys
from pathlib import Path

from tests.test_nodes import _entry_node_list
from tests.test_workflows import _widget_inputs

ROOT = Path(__file__).resolve().parents[1]

# Native (comfy-core) node types the blueprints may chain into; everything else must be ours.
NATIVE_NODE_TYPES = {
    "LoadImage", "LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo", "SaveWEBM",
    "PreviewImage", "SaveImage", "ImageToMask", "MarkdownNote",
}


def _subgraph_files():
    return sorted((ROOT / "subgraphs").glob("*.json"))


def test_subgraphs_exist():
    names = {p.stem for p in _subgraph_files()}
    assert "Image Depth Estimation (Depth Pro 2)" in names
    assert "Video Depth Estimation (Depth Pro 2)" in names


def _internal_node(node_id, g):
    for n in g["nodes"]:
        if n["id"] == node_id:
            return n
    raise AssertionError(f"internal node {node_id} not found")


def test_subgraphs_structure():
    node_classes = {cls.GET_SCHEMA().node_id: cls for cls in _entry_node_list()}
    for path in _subgraph_files():
        doc = json.loads(path.read_text())
        # Top level: the single wrapper node, nothing else.
        assert len(doc["nodes"]) == 1 and doc["links"] == []
        wrapper = doc["nodes"][0]
        assert doc["last_node_id"] >= wrapper["id"]
        assert len(doc["definitions"]["subgraphs"]) == 1
        g = doc["definitions"]["subgraphs"][0]
        assert wrapper["type"] == g["id"]
        assert wrapper["title"] == g["name"] == path.stem
        assert g["category"] == "Preprocessors/Depth"
        assert g["description"]
        assert g["inputNode"]["id"] == -10 and g["outputNode"]["id"] == -20

        # linkIds reference link ids, not positions in the array.
        links = {l["id"]: l for l in g["links"]}

        # The wrapper's inputs/outputs mirror the definition's, in order.
        assert [i["name"] for i in wrapper["inputs"]] == [i["name"] for i in g["inputs"]]
        for w_in, d_in in zip(wrapper["inputs"], g["inputs"]):
            assert w_in["type"] == d_in["type"]
            assert ("widget" in w_in) == (len(d_in["linkIds"]) == 1 and
                                          links[d_in["linkIds"][0]]["origin_id"] == -10
                                          and _is_widget_socket(_internal_node(links[d_in["linkIds"][0]]["target_id"], g),
                                                               links[d_in["linkIds"][0]]["target_slot"]))
        assert [o["name"] for o in wrapper["outputs"]] == [o["name"] for o in g["outputs"]]
        for w_out, d_out in zip(wrapper["outputs"], g["outputs"]):
            assert w_out["type"] == d_out["type"] and w_out["links"] == []

        # Links: unique ids, counters ahead of them.
        assert len(links) == len(g["links"])
        assert g["state"]["lastLinkId"] > max(links)
        assert g["state"]["lastNodeId"] > max(n["id"] for n in g["nodes"])

        by_id = {n["id"]: n for n in g["nodes"]}
        for lid, l in links.items():
            # Endpoint sockets resolve and carry matching types.
            if l["target_id"] == -20:
                # An internal node (or, in a pass-through, the input node) feeds the output.
                d_out = g["outputs"][l["target_slot"]]
                assert lid in d_out["linkIds"] and d_out["type"] == l["type"]
                if l["origin_id"] in by_id:
                    source = by_id[l["origin_id"]]
                    sock = source["outputs"][l["origin_slot"]]
                    assert lid in (sock["links"] or [])
                    assert sock["type"] == l["type"]
            elif l["origin_id"] == -10:
                target = _internal_node(l["target_id"], g)
                sock = target["inputs"][l["target_slot"]]
                assert sock["link"] == lid
                assert sock["type"] == l["type"]
                # A -10 input is either a widget proxy or a plain link socket.
            elif l["origin_id"] in by_id:
                source = by_id[l["origin_id"]]
                sock = source["outputs"][l["origin_slot"]]
                assert lid in (sock["links"] or [])
                target = by_id[l["target_id"]]
                tsock = target["inputs"][l["target_slot"]]
                assert tsock["type"] == l["type"] == sock["type"]
                assert tsock["link"] == lid
            else:
                raise AssertionError(
                    f"link {lid}: unknown origin {l['origin_id']} / target {l['target_id']}")

        # Every definition input/output is wired through its proxy link exactly once.
        for d_in in g["inputs"]:
            l = links[d_in["linkIds"][0]]
            assert l["origin_id"] == -10 and l["origin_slot"] == g["inputs"].index(d_in)
        for d_out in g["outputs"]:
            l = links[d_out["linkIds"][0]]
            assert l["target_id"] == -20 and l["target_slot"] == g["outputs"].index(d_out)

        # proxyWidgets line up with the wrapper's widget inputs, in order.
        widget_indexes = [i for i, w_in in enumerate(wrapper["inputs"]) if "widget" in w_in]
        proxy = wrapper["properties"]["proxyWidgets"]
        assert len(proxy) == len(widget_indexes)
        for pos, (w_idx, (node_id, widget_name)) in enumerate(zip(widget_indexes, proxy)):
            l = links[g["inputs"][w_idx]["linkIds"][0]]
            target = _internal_node(int(node_id), g)
            sock = target["inputs"][l["target_slot"]]
            assert sock["widget"] == {"name": widget_name} == {"name": wrapper["inputs"][w_idx]["widget"]["name"]}
            assert pos == widget_indexes.index(w_idx)

        # Internal nodes: known types, sockets and widgets match the live V3 schemas.
        for n in g["nodes"]:
            cls = node_classes.get(n["type"])
            assert cls is not None or n["type"] in NATIVE_NODE_TYPES, n["type"]
            if cls is None:
                continue
            schema = cls.GET_SCHEMA()
            sockets = [(i.id, i.io_type, type(i) in _widget_types()) for i in schema.inputs]
            assert len(sockets) == len(n["inputs"]), (n["type"], n["inputs"])
            for slot, sock in enumerate(n["inputs"]):
                inp_id, inp_type, is_widget = sockets[slot]
                assert sock["name"] == inp_id and sock["type"] == inp_type, (n["type"], slot)
                assert ("widget" in sock) == is_widget, (n["type"], sock["name"])
            out_sockets = [o.id for o in schema.outputs]
            assert [o["name"] for o in n["outputs"]] == out_sockets, (n["type"], n["outputs"])
            widget_count = sum(1 for s in sockets if s[2])
            assert len(n["widgets_values"]) == widget_count, (n["type"], n["widgets_values"])


def _is_widget_socket(node, slot):
    return "widget" in node["inputs"][slot]


def _widget_types():
    from comfy_api.latest import io

    return (io.Float.Input, io.Int.Input, io.String.Input, io.Boolean.Input, io.Combo.Input)
