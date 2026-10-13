"""Pipelines that enhance the prompt must sample the enhancement with the request seed."""

import ast
import importlib
import inspect

import pytest

SEEDED_PIPELINE_MODULES = [
    "ltx_pipelines.ic_lora",
    "ltx_pipelines.keyframe_interpolation",
    "ltx_pipelines.retake",
    "ltx_pipelines.t2a_one_stage",
    "ltx_pipelines.ti2vid_one_stage",
    "ltx_pipelines.ti2vid_two_stages",
    "ltx_pipelines.ti2vid_two_stages_hq",
    "ltx_pipelines.dubit",
]


def _enhancing_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and any(kw.arg == "enhance_first_prompt" for kw in node.keywords)
    ]


@pytest.mark.parametrize("module_name", SEEDED_PIPELINE_MODULES)
def test_enhancement_is_seeded_with_the_request_seed(module_name: str) -> None:
    tree = ast.parse(inspect.getsource(importlib.import_module(module_name)))
    calls = _enhancing_calls(tree)
    assert calls, f"{module_name} no longer calls the prompt encoder with enhance_first_prompt"
    for call in calls:
        seed = next((kw.value for kw in call.keywords if kw.arg == "enhance_prompt_seed"), None)
        assert isinstance(seed, ast.Name), f"{module_name}:{call.lineno} does not pass enhance_prompt_seed"
        assert seed.id == "seed", f"{module_name}:{call.lineno} passes enhance_prompt_seed={seed.id}, not seed"
