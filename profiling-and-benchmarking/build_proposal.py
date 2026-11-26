"""Build two complete, matching SciPy checkouts with the upstream spin workflow."""

import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from packaging.requirements import Requirement

import experiment as ex

REQUIREMENTS = ("meson-python>=0.15", "meson>=1.5", "Cython>=3.2",
                "pybind11>=2.13.2", "pythran>=0.18.1", "numpy>=2.0",
                "spin>=0.14", "pytest>=8", "threadpoolctl", "hypothesis")


def prerequisites():
    problems = []
    if not sys.platform.startswith("linux"):
        problems.append("This recipe currently supports Linux/WSL.")
    for text in REQUIREMENTS:
        requirement = Requirement(text)
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"Missing Python dependency: {text}")
        else:
            if not requirement.specifier.contains(version):
                problems.append(f"Need {text}; installed {version}")
    for tool in ("git", "patch", "meson", "ninja", "pkg-config"):
        if shutil.which(tool) is None:
            problems.append(f"Missing executable: {tool}; activate the build environment.")
    for variant in ex.VARIANTS:
        source = ex.source_dir(variant)
        if not (source / ".git").exists():
            problems.append(f"Prepare the pinned {variant} checkout: {source}")
    return problems


def git(source, *args):
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(source), *args], text=True).strip()


def prepare(variant):
    source = ex.source_dir(variant)
    spec = ex.MANIFEST["scipy"]
    if git(source, "rev-parse", "HEAD") != spec["commit"]:
        raise RuntimeError(f"{source} must be at commit {spec['commit']}")
    modules = git(source, "submodule", "status", "--recursive")
    # strip() removes the first normal leading space; inspect each status by its
    # non-space prefix so an uninitialized or mismatched module cannot pass.
    if any(line.lstrip().startswith(("-", "+", "U")) for line in modules.splitlines()):
        raise RuntimeError(f"Initialize the pinned submodules in {source} before building")
    changed = set(git(source, "diff", "--name-only", spec["commit"]).splitlines())
    allowed = {spec["core"]} if variant == "proposal" else set()
    if changed - allowed:
        raise RuntimeError(f"Preserve manual edits before using this checkout: {changed}")
    core = source / spec["core"]
    normalized = hashlib.sha256(core.read_text().encode()).hexdigest()
    if variant == "proposal" and normalized == spec["baseline_sha256"]:
        result = subprocess.run(["patch", "--batch", "--fuzz=0", str(core)],
                                input=ex.PATCH.read_text(), capture_output=True, text=True,
                                cwd=source, check=True)
        (ex.BUILD / "patch.log").write_text(result.stdout + result.stderr)
        normalized = hashlib.sha256(core.read_text().encode()).hexdigest()
    if normalized != spec[f"{variant}_sha256"]:
        raise RuntimeError(f"Unexpected {variant} source; no overwrite attempted: {core}")
    return {"commit": spec["commit"], "core_sha256": normalized, "submodules": modules}


def direction_body(source):
    start = source.index("            CBLAS_INT pointr;\n", source.index("void\nmainlb("))
    end = source.index("            iword = 0;\n", start)
    return "    CBLAS_INT i, one_int = 1;\n" + "".join(
        line[8:] if line.strip() else line for line in source[start:end].splitlines(keepends=True))


def build_kernel():
    """Compile the proposal's exact recursion body for independent dense checks."""
    source = ex.source_dir("proposal")
    body = direction_body((source / ex.MANIFEST["scipy"]["core"]).read_text())
    bridge = ex.BUILD / "direction.c"
    bridge.write_text(
        '#include "lbfgsb.h"\n'
        '__attribute__((visibility("default"))) void proposal_direction(\n'
        "    CBLAS_INT n, CBLAS_INT m, double* x, double* g, double* z,\n"
        "    double* ws, double* wy, double* sy, double theta, CBLAS_INT col,\n"
        "    CBLAS_INT head, double* wa)\n{\n" + body + "}\n")
    pkg = shlex.split(subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", "openblas"], text=True))
    command = (shlex.split(os.environ.get("CC", "cc"))
               + shlex.split(os.environ.get("CFLAGS", ""))
               + ["-std=c17", "-O2", "-fPIC", "-shared", str(bridge),
                  f"-I{source}/scipy/optimize/src", f"-I{source}/scipy/_build_utils/src"]
               + shlex.split(os.environ.get("LDFLAGS", "")) + pkg
               + ["-o", str(ex.BUILD / "direction.so")])
    run_logged(command, ex.HERE, ex.BUILD / "kernel.log")
    return command


def run_logged(command, directory, log):
    scratch = ex.BUILD / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TMPDIR"] = str(scratch)
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=directory, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed; see {log}")


def main():
    problems = prerequisites()
    if problems:
        raise RuntimeError("Build prerequisites are not ready:\n- " + "\n- ".join(problems)
                           + "\nFollow ENV_MEMO.md; this script does not install or clone anything.")
    if ex.digest(ex.PATCH) != ex.MANIFEST["patch"]["sha256"]:
        raise RuntimeError("The proposal patch differs from manifest.json")
    sources = {name: prepare(name) for name in ex.VARIANTS}
    commands = []
    for name in ex.VARIANTS:
        # Upstream workflow: https://scipy.github.io/devdocs/dev/building/contributor.html
        command = [sys.executable, "-m", "spin", "build", "--build-dir=build",
                   "-j", str(ex.MANIFEST["build"]["jobs"])]
        command += [f"--setup-args={arg}" for arg in ex.MANIFEST["build"]["setup_args"]]
        print(f"Building complete SciPy: {name}", flush=True)
        run_logged(command, ex.source_dir(name), ex.BUILD / f"{name}-build.log")
        commands.append(command)
        ex.installed_site(name)
    commands.append(build_kernel())
    meson = {}
    for name in ex.VARIANTS:
        meson[name] = {}
        for kind in ("compilers", "buildoptions"):
            command = ["meson", "introspect", f"--{kind}", str(ex.source_dir(name) / "build")]
            meson[name][kind] = json.loads(subprocess.check_output(command, text=True))
    info = {"prepared_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": ex.MANIFEST, "patch_sha256": ex.digest(ex.PATCH),
            "python_executable": sys.executable, "python": sys.version,
            "python_version": list(sys.version_info[:2]),
            "source_records": sources, "commands": commands, "meson": meson,
            "python_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "kernel_sha256": ex.digest(ex.BUILD / "direction.so"),
            "extension_sha256": {name: ex.digest(next(
                (ex.installed_site(name) / "scipy/optimize").glob("_lbfgsb*.so")))
                for name in ex.VARIANTS}}
    ex.RESULTS.mkdir(parents=True, exist_ok=True)
    (ex.RESULTS / "build.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
