"""One-off generator for subgraphs/*.json (the ComfyUI "blueprints" served at /global_subgraphs).

Re-run after changing node schemas; tests/test_subgraphs.py keeps the committed JSON honest.
"""

import json
import uuid
from pathlib import Path


def uid():
    return str(uuid.uuid4())


def node(node_id, node_type, pos, size, order, inputs, outputs, widgets, cnr="ComfyUI-DepthPro2"):
    return {
        "id": node_id, "type": node_type, "pos": pos, "size": size,
        "flags": {}, "order": order, "mode": 0,
        "inputs": inputs, "outputs": outputs,
        "properties": {
            "Node name for S&R": node_type,
            "cnr_id": cnr,
            "enableTabs": False, "tabWidth": 65, "tabXOffset": 10,
            "hasSecondTab": False, "secondTabText": "Send Back",
            "secondTabOffset": 80, "secondTabWidth": 65,
        },
        "widgets_values": widgets,
    }


def sock_in(name, type_, link=None, widget=False):
    d = {"name": name, "type": type_, "link": link}
    if widget:
        d["widget"] = {"name": name}
    d["localized_name"] = name
    return d


def sock_out(name, type_, links):
    return {"name": name, "type": type_, "links": links, "localized_name": name}


def link(id, o, os_, t, ts, type_):
    return {"id": id, "origin_id": o, "origin_slot": os_, "target_id": t, "target_slot": ts, "type": type_}


def def_in(name, type_, link_id, pos):
    return {"id": uid(), "name": name, "type": type_, "linkIds": [link_id], "localized_name": name, "pos": pos}


def def_out(name, type_, link_id, pos):
    return {"id": uid(), "name": name, "type": type_, "linkIds": [link_id], "localized_name": name, "pos": pos}


# ---------------------------------------------------------------- image variant
IMG_NAME = "Image Depth Estimation (Depth Pro 2)"
IN_X, IN_Y = -1000.0, -140.0
OUT_X, OUT_Y = 640.0, -100.0

loader = node(2, "DepthPro2Loader", [-760, -140], [270, 140], 0,
    [sock_in("model", "COMBO", 102, widget=True),
     sock_in("precision", "COMBO", widget=True),
     sock_in("device", "COMBO", widget=True)],
    [sock_out("model", "DEPTHPRO2_MODEL", [101])],
    ["depthpro2-s", "auto", "auto"])
estimate = node(4, "DepthPro2Estimate", [-420, -180], [270, 180], 1,
    [sock_in("model", "DEPTHPRO2_MODEL", 101),
     sock_in("image", "IMAGE", 103),
     sock_in("focal_length_mm", "FLOAT", 104, widget=True),
     sock_in("max_size", "INT", 105, widget=True),
     sock_in("focal_length_px", "FLOAT", widget=True)],
    [sock_out("depth_image", "IMAGE", [107]),
     sock_out("depth_meters", "DEPTHPRO2_DEPTH", [106, 109]),
     sock_out("focal_length_px", "FLOAT", []),
     sock_out("sky_mask", "MASK", [110])],
    [0.0, 0, 0.0])
colorize = node(7, "DepthPro2Colorize", [-80, -100], [270, 80], 2,
    [sock_in("depth", "DEPTHPRO2_DEPTH", 106),
     sock_in("colormap", "COMBO", widget=True)],
    [sock_out("depth_color", "IMAGE", [108])],
    ["turbo"])

img_inputs = [
    def_in("model", "COMBO", 102, [IN_X + 143, IN_Y + 24]),
    def_in("source_image", "IMAGE", 103, [IN_X + 143, IN_Y + 44]),
    def_in("focal_length_mm", "FLOAT", 104, [IN_X + 143, IN_Y + 64]),
    def_in("max_size", "INT", 105, [IN_X + 143, IN_Y + 84]),
]
img_outputs = [
    def_out("depth_image", "IMAGE", 107, [OUT_X + 24, OUT_Y + 24]),
    def_out("depth_color", "IMAGE", 108, [OUT_X + 24, OUT_Y + 44]),
    def_out("depth_meters", "DEPTHPRO2_DEPTH", 109, [OUT_X + 24, OUT_Y + 64]),
    def_out("sky_mask", "MASK", 110, [OUT_X + 24, OUT_Y + 84]),
]
img_links = [
    link(101, 2, 0, 4, 0, "DEPTHPRO2_MODEL"),
    link(102, -10, 0, 2, 0, "COMBO"),
    link(103, -10, 1, 4, 1, "IMAGE"),
    link(104, -10, 2, 4, 2, "FLOAT"),
    link(105, -10, 3, 4, 3, "INT"),
    link(106, 4, 1, 7, 0, "DEPTHPRO2_DEPTH"),
    link(107, 4, 0, -20, 0, "IMAGE"),
    link(108, 7, 0, -20, 1, "IMAGE"),
    link(109, 4, 1, -20, 2, "DEPTHPRO2_DEPTH"),
    link(110, 4, 3, -20, 3, "MASK"),
]
img_wrapper = {
    "inputs": [
        {"localized_name": "model", "name": "model", "type": "COMBO", "widget": {"name": "model"}, "link": None},
        {"localized_name": "source_image", "name": "source_image", "type": "IMAGE", "link": None},
        {"localized_name": "focal_length_mm", "name": "focal_length_mm", "type": "FLOAT", "widget": {"name": "focal_length_mm"}, "link": None},
        {"localized_name": "max_size", "name": "max_size", "type": "INT", "widget": {"name": "max_size"}, "link": None},
    ],
    "outputs": [
        {"localized_name": n, "name": n, "type": t, "links": []}
        for n, t in [("depth_image", "IMAGE"), ("depth_color", "IMAGE"),
                     ("depth_meters", "DEPTHPRO2_DEPTH"), ("sky_mask", "MASK")]
    ],
    "proxyWidgets": [["2", "model"], ["4", "focal_length_mm"], ["4", "max_size"]],
}

img = {
    "revision": 0, "last_node_id": 100, "last_link_id": 0,
    "nodes": [None], "links": [], "version": 0.4,
    "definitions": {"subgraphs": [{
        "id": uid(), "version": 1,
        "state": {"lastGroupId": 0, "lastNodeId": 8, "lastLinkId": 111, "lastRerouteId": 0},
        "revision": 0, "config": {}, "name": IMG_NAME,
        "inputNode": {"id": -10, "bounding": [IN_X, IN_Y, 167.3, 148]},
        "outputNode": {"id": -20, "bounding": [OUT_X, OUT_Y, 167.3, 148]},
        "inputs": img_inputs, "outputs": img_outputs, "widgets": [],
        "nodes": [loader, estimate, colorize], "groups": [], "links": img_links,
        "extra": {}, "category": "Preprocessors/Depth",
        "description": ("Estimates metric depth in meters from an input image using Apple Depth Pro 2, "
                        "outputting a normalized preview, a colorized map, raw metric depth and a sky mask."),
    }]},
    "extra": {},
}

# ---------------------------------------------------------------- video variant
VID_NAME = "Video Depth Estimation (Depth Pro 2)"
IN_X, IN_Y = -1200.0, -80.0
OUT_X, OUT_Y = 320.0, -80.0

loader_v = node(2, "DepthPro2Loader", [-960, 100], [270, 140], 0,
    [sock_in("model", "COMBO", 202, widget=True),
     sock_in("precision", "COMBO", widget=True),
     sock_in("device", "COMBO", widget=True)],
    [sock_out("model", "DEPTHPRO2_MODEL", [203])],
    ["depthpro2-s", "auto", "auto"])
getvideo = node(4, "GetVideoComponents", [-960, -60], [270, 100], 1,
    [sock_in("video", "VIDEO", 201)],
    [sock_out("images", "IMAGE", [204]),
     sock_out("audio", "AUDIO", []),
     sock_out("fps", "FLOAT", [208]),
     sock_out("bit_depth", "COMBO", []),
     sock_out("color_space", "COMBO", [])],
    [], cnr="comfy-core")
video = node(5, "DepthPro2Video", [-540, -140], [270, 220], 2,
    [sock_in("model", "DEPTHPRO2_MODEL", 203),
     sock_in("images", "IMAGE", 204),
     sock_in("window", "INT", 205, widget=True),
     sock_in("overlap", "INT", widget=True),
     sock_in("focal_length_mm", "FLOAT", 206, widget=True),
     sock_in("max_size", "INT", 207, widget=True),
     sock_in("mode", "COMBO", widget=True),
     sock_in("focal_length_px", "FLOAT", widget=True)],
    [sock_out("depth_image", "IMAGE", [209, 210]),
     sock_out("depth_meters", "DEPTHPRO2_DEPTH", [212]),
     sock_out("focal_length_px", "FLOAT", []),
     sock_out("sky_mask", "MASK", [213])],
    [16, 4, 0.0, 0, "buffered", 0.0])
createvideo = node(6, "CreateVideo", [-160, -60], [270, 180], 3,
    [sock_in("images", "IMAGE", 209),
     sock_in("fps", "FLOAT", 208, widget=True),
     sock_in("audio", "AUDIO"),
     sock_in("bit_depth", "COMBO", widget=True),
     sock_in("color_space", "COMBO", widget=True),
     sock_in("chroma_subsampling", "COMBO", widget=True),
     sock_in("codec", "COMBO", widget=True)],
    [sock_out("video", "VIDEO", [211])],
    [30.0, "auto", "sRGB", "auto", "none"], cnr="comfy-core")

vid_inputs = [
    def_in("video", "VIDEO", 201, [IN_X + 143, IN_Y + 24]),
    def_in("model", "COMBO", 202, [IN_X + 143, IN_Y + 44]),
    def_in("window", "INT", 205, [IN_X + 143, IN_Y + 64]),
    def_in("focal_length_mm", "FLOAT", 206, [IN_X + 143, IN_Y + 84]),
    def_in("max_size", "INT", 207, [IN_X + 143, IN_Y + 104]),
]
vid_outputs = [
    def_out("depth_image", "IMAGE", 210, [OUT_X + 24, OUT_Y + 24]),
    def_out("depth_video", "VIDEO", 211, [OUT_X + 24, OUT_Y + 44]),
    def_out("depth_meters", "DEPTHPRO2_DEPTH", 212, [OUT_X + 24, OUT_Y + 64]),
    def_out("sky_mask", "MASK", 213, [OUT_X + 24, OUT_Y + 84]),
]
vid_links = [
    link(201, -10, 0, 4, 0, "VIDEO"),
    link(202, -10, 1, 2, 0, "COMBO"),
    link(203, 2, 0, 5, 0, "DEPTHPRO2_MODEL"),
    link(204, 4, 0, 5, 1, "IMAGE"),
    link(205, -10, 2, 5, 2, "INT"),
    link(206, -10, 3, 5, 4, "FLOAT"),
    link(207, -10, 4, 5, 5, "INT"),
    link(208, 4, 2, 6, 1, "FLOAT"),
    link(209, 5, 0, 6, 0, "IMAGE"),
    link(210, 5, 0, -20, 0, "IMAGE"),
    link(211, 6, 0, -20, 1, "VIDEO"),
    link(212, 5, 1, -20, 2, "DEPTHPRO2_DEPTH"),
    link(213, 5, 3, -20, 3, "MASK"),
]
vid_wrapper = {
    "inputs": [
        {"localized_name": "video", "name": "video", "type": "VIDEO", "link": None},
        {"localized_name": "model", "name": "model", "type": "COMBO", "widget": {"name": "model"}, "link": None},
        {"localized_name": "window", "name": "window", "type": "INT", "widget": {"name": "window"}, "link": None},
        {"localized_name": "focal_length_mm", "name": "focal_length_mm", "type": "FLOAT", "widget": {"name": "focal_length_mm"}, "link": None},
        {"localized_name": "max_size", "name": "max_size", "type": "INT", "widget": {"name": "max_size"}, "link": None},
    ],
    "outputs": [
        {"localized_name": n, "name": n, "type": t, "links": []}
        for n, t in [("depth_image", "IMAGE"), ("depth_video", "VIDEO"),
                     ("depth_meters", "DEPTHPRO2_DEPTH"), ("sky_mask", "MASK")]
    ],
    "proxyWidgets": [["2", "model"], ["5", "window"], ["5", "focal_length_mm"], ["5", "max_size"]],
}

vid = {
    "revision": 0, "last_node_id": 200, "last_link_id": 0,
    "nodes": [None], "links": [], "version": 0.4,
    "definitions": {"subgraphs": [{
        "id": uid(), "version": 1,
        "state": {"lastGroupId": 0, "lastNodeId": 7, "lastLinkId": 214, "lastRerouteId": 0},
        "revision": 0, "config": {}, "name": VID_NAME,
        "inputNode": {"id": -10, "bounding": [IN_X, IN_Y, 167.3, 168]},
        "outputNode": {"id": -20, "bounding": [OUT_X, OUT_Y, 167.3, 148]},
        "inputs": vid_inputs, "outputs": vid_outputs, "widgets": [],
        "nodes": [loader_v, getvideo, video, createvideo], "groups": [], "links": vid_links,
        "extra": {}, "category": "Preprocessors/Depth",
        "description": ("Estimates temporally consistent metric depth in meters from an input video using "
                        "Apple Depth Pro 2, outputting preview frames, a depth video, raw metric depth and a sky mask."),
    }]},
    "extra": {},
}


def finalize(doc, wrapper_id, pos, wrapper):
    g = doc["definitions"]["subgraphs"][0]
    doc["nodes"][0] = {
        "id": wrapper_id, "type": g["id"], "pos": pos, "size": [430, 330],
        "flags": {}, "order": 0, "mode": 0,
        "inputs": wrapper["inputs"], "outputs": wrapper["outputs"],
        "properties": {
            "proxyWidgets": wrapper["proxyWidgets"],
            "cnr_id": "ComfyUI-DepthPro2",
            "enableTabs": False, "tabWidth": 65, "tabXOffset": 10,
            "hasSecondTab": False, "secondTabText": "Send Back",
            "secondTabOffset": 80, "secondTabWidth": 65,
        },
        "widgets_values": [], "title": g["name"],
    }
    return doc


img = finalize(img, 100, [-1000, -320], img_wrapper)
vid = finalize(vid, 200, [-1200, -300], vid_wrapper)

out = Path(__file__).resolve().parents[1] / "subgraphs"
out.mkdir(exist_ok=True)
for name, doc in ((IMG_NAME, img), (VID_NAME, vid)):
    path = out / f"{name}.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote", path)
