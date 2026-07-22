import sys
from dataclasses import make_dataclass
from types import ModuleType
import unittest
from unittest.mock import patch

from vortex_torch.engine.sgl.config import (
    _REQUIRED_SGLANG_HOOKS,
    VORTEX_SGLANG_ABI,
    VORTEX_SGLANG_PLUGIN_TARGETS,
    validate_sglang_runtime_contract,
)


def _fake_plugin_sglang(*, abi=VORTEX_SGLANG_ABI, hooks=(), fields=()):
    sglang = ModuleType("sglang")
    srt = ModuleType("sglang.srt")
    plugins = ModuleType("sglang.srt.plugins")
    hook_registry = ModuleType("sglang.srt.plugins.hook_registry")
    server_args = ModuleType("sglang.srt.server_args")

    server_args.ServerArgs = make_dataclass("ServerArgs", fields)
    server_args.ServerArgs._vortex_sglang_abi = abi
    server_args.ServerArgs._vortex_sglang_hooks = frozenset(hooks)
    server_args.ServerArgs._vortex_sglang_targets = VORTEX_SGLANG_PLUGIN_TARGETS
    hook_registry.HookRegistry = type(
        "HookRegistry", (), {"_patched": set(VORTEX_SGLANG_PLUGIN_TARGETS)}
    )
    plugins.load_plugins = lambda: None
    sglang.srt = srt

    return patch.dict(
        sys.modules,
        {
            "sglang": sglang,
            "sglang.srt": srt,
            "sglang.srt.plugins": plugins,
            "sglang.srt.plugins.hook_registry": hook_registry,
            "sglang.srt.server_args": server_args,
        },
    )


class SGLangRuntimeContractTest(unittest.TestCase):
    def test_contract_accepts_complete_plugin_runtime(self):
        with _fake_plugin_sglang(
            hooks=_REQUIRED_SGLANG_HOOKS,
            fields=[("vortex", object)],
        ):
            validate_sglang_runtime_contract()

    def test_contract_rejects_incomplete_runtime(self):
        with _fake_plugin_sglang(abi=None):
            with self.assertRaisesRegex(
                RuntimeError, "Incompatible official SGLang plugin runtime"
            ):
                validate_sglang_runtime_contract()


if __name__ == "__main__":
    unittest.main()
