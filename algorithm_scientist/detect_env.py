"""Detect (and recommend) a working environment for vortex_torch.

The workflow must NOT assume a specific conda env exists. This probes what's on
the host — conda envs, uv, venvs, docker, the current interpreter — finds where
`import vortex_torch` actually works, and prints the **run prefix** the agent
should prepend to every python call this session (plus a separate one for GLM,
which needs a newer transformers). If nothing works, it reports what's available
so `/setup-env` can build one.

Robust to: no conda, no `vortex_v1`, uv/venv/docker setups, GLM's transformers
split. Stdlib only; probes are subprocesses with timeouts.

Usage
-----
::

    python algorithm_scientist/detect_env.py            # full probe + recommendation
    python algorithm_scientist/detect_env.py --json
    python algorithm_scientist/detect_env.py --no-import # fast: tooling/envs only
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# What we want to know about a candidate interpreter.
PROBE = (
    "import json\n"
    "out={}\n"
    "try:\n"
    "    import vortex_torch as v; out['vortex_torch']=getattr(v,'__version__','?')\n"
    "except Exception as e:\n"
    "    out['vortex_torch']=None; out['vortex_err']=str(e)[:200]\n"
    "for m in ('torch','transformers','triton'):\n"
    "    try:\n"
    "        out[m]=__import__(m).__version__\n"
    "    except Exception:\n"
    "        out[m]=None\n"
    "try:\n"
    "    import sglang; out['sglang']=getattr(sglang,'__version__','?')\n"
    "except Exception:\n"
    "    out['sglang']=None\n"
    "import sys; out['exe']=sys.executable\n"
    "print(json.dumps(out))\n"
)


def _run(cmd, timeout):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 1, "", str(e)[:200]


def _conda_base():
    rc, out, _ = _run(["conda", "info", "--base"], 20)
    return out if rc == 0 and out else None


def _conda_envs():
    rc, out, _ = _run(["conda", "env", "list"], 20)
    envs = []
    if rc == 0:
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split()[0]
            if name and name != "*":
                envs.append(name)
    return envs


def detect_tooling():
    return {
        "conda": shutil.which("conda"),
        "conda_base": _conda_base() if shutil.which("conda") else None,
        "uv": shutil.which("uv"),
        "docker": shutil.which("docker"),
        "pip": shutil.which("pip") or shutil.which("pip3"),
        "python": sys.executable,
        "repo_specs": [f for f in ("environment.yml", "requirements.txt",
                                   "pyproject.toml", "setup.py", "Dockerfile")
                       if (REPO / f).exists()],
    }


def candidates(tooling):
    """Ordered (label, command-prefix-list) to probe for `import vortex_torch`."""
    cands = []
    # conda envs whose name suggests vortex (vortex_v1 default, then others)
    if tooling["conda"]:
        envs = _conda_envs()
        for e in sorted(envs, key=lambda x: (x != "vortex_v1", "vortex" not in x, x)):
            if "vortex" in e or e in ("base",):
                cands.append((f"conda:{e}", ["conda", "run", "-n", e, "python"]))
    # repo venvs
    for vn in (".venv", "venv"):
        py = REPO / vn / "bin" / "python"
        if py.exists():
            cands.append((f"venv:{vn}", [str(py)]))
    # current interpreter
    cands.append(("current", [sys.executable]))
    # uv (project-managed)
    if tooling["uv"] and (REPO / "pyproject.toml").exists():
        cands.append(("uv", ["uv", "run", "python"]))
    return cands


def probe(cands, timeout):
    results = []
    for label, prefix in cands:
        rc, out, err = _run(prefix + ["-c", PROBE], timeout)
        info = {"label": label, "prefix": " ".join(prefix)}
        try:
            info.update(json.loads(out.splitlines()[-1]))
        except Exception:
            info["vortex_torch"] = None
            info["vortex_err"] = (err or out)[:200]
        results.append(info)
    return results


def _tf_major(v):
    try:
        return int(str(v).split(".")[0])
    except Exception:
        return 0


def recommend(results):
    ok = [r for r in results if r.get("vortex_torch")]
    # default vortex env: importable + has sglang (the serving stack)
    default = next((r for r in ok if r.get("sglang")), (ok[0] if ok else None))
    # GLM env: importable + transformers major >= 5
    glm = next((r for r in ok if _tf_major(r.get("transformers")) >= 5), None)
    return default, glm


def main():
    ap = argparse.ArgumentParser(description="Detect a working vortex_torch environment.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-import", action="store_true", help="skip import probes (fast)")
    ap.add_argument("--timeout", type=int, default=120, help="per-candidate probe timeout (s)")
    args = ap.parse_args()

    tooling = detect_tooling()
    results = [] if args.no_import else probe(candidates(tooling), args.timeout)
    default, glm = recommend(results)

    if args.json:
        print(json.dumps({"tooling": tooling, "results": results,
                          "recommended_default": default, "recommended_glm": glm}, indent=2))
        return

    print("# vortex_torch environment detection\n")
    print("Tooling: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in
          (("conda", tooling["conda"]), ("uv", tooling["uv"]),
           ("docker", tooling["docker"]), ("pip", tooling["pip"]))))
    print(f"  conda base: {tooling['conda_base'] or '—'}")
    print(f"  repo env specs: {', '.join(tooling['repo_specs']) or '—'}")
    if args.no_import:
        print("\n(--no-import: skipped probes)")
        return
    print("\nProbed interpreters (does `import vortex_torch` work?):")
    for r in results:
        v = r.get("vortex_torch")
        tag = f"OK v={v}" if v else f"NO ({r.get('vortex_err','')[:60]})"
        print(f"  [{tag}] {r['label']:<16} torch={r.get('torch')} "
              f"transformers={r.get('transformers')} sglang={r.get('sglang')}")
    print()
    if default:
        print(f"** RECOMMENDED (default vortex): `{default['prefix']}` "
              f"(transformers {default.get('transformers')}, sglang {default.get('sglang')})")
        print(f"   use it as a run prefix, e.g.: {default['prefix']} algorithm_scientist/run_submission.py ...")
    else:
        print("** No working vortex_torch env found — build one (see /setup-env): "
              "create from repo specs (" + (", ".join(tooling["repo_specs"]) or "none") +
              ") via conda/uv/venv/docker, then `pip install -e .`.")
    if glm and glm is not default:
        print(f"** GLM env (transformers>=5): `{glm['prefix']}` (transformers {glm.get('transformers')})")
    elif not glm:
        print("** GLM (glm4_moe*) needs transformers>=5 — none detected; /setup-env can build a separate env.")


if __name__ == "__main__":
    main()
