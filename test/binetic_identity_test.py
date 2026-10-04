"""Test Binetic identity injection end-to-end."""
import json
import os
import sys

sys.path.insert(0, "/Users/binetic/TPN-model")

from tpn_engine.binetic_converter import BineticConverter
from tpn_engine.binetic import BineticFormat

model_data = {
    "architecture": "transformer",
    "model_name": "tiny-tpn-demo",
    "layers": [
        {"name": "layer_0", "type": "transformer",
         "weights": [[0.1, 0.2], [0.3, 0.4]]},
        {"name": "layer_1", "type": "transformer",
         "weights": [[0.5, 0.6], [0.7, 0.8]]},
    ],
    "connections": [("layer_0", "layer_1")],
}

conv = BineticConverter()
graph = conv.convert(model_data)
assert graph.is_binetic(), "graph must be Binetic"
assert graph.identity.owner == "binetic.ai"
assert graph.identity.tool == "binetic_converter"
assert graph.identity.derived_from == "tiny-tpn-demo"
assert "Max Yeremenko" in graph.identity.partners
assert "__binetic__" in graph.metadata
assert "binetic" in graph.metadata

fmt = BineticFormat()
out = "/tmp/demo.binetic"
fmt.save(out, graph, version_id="v1")

g2, v2 = fmt.load(out)
assert g2.is_binetic(), "loaded graph must be Binetic"
assert g2.identity.tool == "binetic_converter"
assert g2.identity.derived_from == "tiny-tpn-demo"

ident = fmt.identity_of(out)
assert ident is not None
assert ident.owner == "binetic.ai"

# Header identity matches graph identity
with open(out) as f:
    hdr = json.load(f)
assert hdr["identity"]["owner"] == "binetic.ai"
print("ALL OK")
