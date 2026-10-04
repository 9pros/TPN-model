"""Test TPNModel Binetic persistence and identity."""
import sys
sys.path.insert(0, "/Users/binetic/TPN-model")

from tpn_engine import TPNConfig, TPNModel
from tpn_engine.binetic import BineticIdentity

cfg = TPNConfig(hidden_size=16, num_layers=2, intermediate_size=32,
                vocab_size=32, max_position_embeddings=128)
model = TPNModel(cfg)

print("before save - is_binetic:", model.is_binetic())
print("num params:", model.num_parameters())

# Save to binetic
out = "/tmp/model.binetic"
model.save_to_binetic(out, version_id="v1")
print("saved", out)

# Load back
graph = model.load_from_binetic(out)
print("after load - is_binetic:", model.is_binetic())
print("identity owner:", model.binetic_identity.owner)
print("identity tool:", model.binetic_identity.tool)
print("identity derived_from:", model.binetic_identity.derived_from)
print("identity partners:", model.binetic_identity.partners)
print("identity gen info:", model.binetic_identity.generation_info)
print("layers in graph:", len(graph.nodes))
print("ALL OK")
