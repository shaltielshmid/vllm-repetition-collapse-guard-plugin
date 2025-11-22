# setup.py
from setuptools import setup

setup(
    name="vllm-repguard",
    version="0.1.0",
    py_modules=["vllm_repguard_plugin"],
    install_requires=[
        "vllm>=0.9.1",   # V1 is default from ~0.8+; adjust if you need older
    ],
    entry_points={
        "vllm.general_plugins": [
            # name "repguard" is arbitrary; only used for logging/filtering
            "repguard = vllm_repguard_plugin:register_repguard",
        ]
    },
    python_requires=">=3.10",
)
