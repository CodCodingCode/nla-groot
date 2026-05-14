"""Natural Language Autoencoders for GR00T N1.7.

Public submodules:
    layer_spec      - canonical architecture constants (target tensor names, dims).
    extraction      - PyTorch forward hooks and activation dump utilities.
    labeling        - OpenAI-backed warm-start labeling pipeline.
    models          - AV (verbalizer) and AR (reconstructor) implementations.
    training        - SFT and GRPO trainers operating on per-token samples.
    steering        - Edit-then-patch harness for causal validation in SimplerEnv.
    eval            - FVE, confabulation, memorization-vs-generalization, etc.
    viz             - Spatial NLA maps, concept timelines, concept clustering.
"""
__version__ = "0.0.1"
