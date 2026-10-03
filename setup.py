[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "tpn-model"
version = "0.1.0"
description = "Temporal Packet Network Inference Engine"
readme = "README.md"
requires-python = ">=3.8"
authors = [
    {name = "TPN Team"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["tpn_engine*"]
