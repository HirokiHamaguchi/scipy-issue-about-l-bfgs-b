"""Build baseline and proposal in scratch space without modifying SciPy."""

from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sysconfig

import experiment as ex


def direction_body(source):
    start = source.index("            CBLAS_INT pointr;\n", source.index("void\nmainlb("))
    end = source.index("            iword = 0;\n", start)
    body = source[start:end]
    return "    CBLAS_INT i, one_int = 1;\n" + "".join(
        line[8:] if line.strip() else line for line in body.splitlines(keepends=True))


def prepare():
    baseline = (ex.SCIPY_SOURCE / "scipy/optimize/src/lbfgsb.c").read_text()
    assert hashlib.sha256(baseline.encode()).hexdigest() == ex.BASELINE_SHA256
    ex.RESULTS.mkdir(parents=True, exist_ok=True)
    for variant in ex.VARIANTS:
        folder = ex.WORK / "sources" / variant
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "lbfgsb.c").write_text(baseline, newline="\n")
    proposal = ex.WORK / "sources/proposal/lbfgsb.c"
    # Apply to an ordinary copied file. No Git index or checkout is changed.
    result = subprocess.run(["patch", "--batch", "--fuzz=0", str(proposal)],
                            input=ex.PATCH.read_text(), capture_output=True, text=True,
                            cwd=ex.WORK, check=True)
    (ex.WORK / "patch.log").write_text(result.stdout + result.stderr)
    patched = proposal.read_text()
    body = direction_body(patched)
    wrapper = (
        '\n__attribute__((visibility("default"))) void proposal_direction(\n'
        "    CBLAS_INT n, CBLAS_INT m, double* x, double* g, double* z,\n"
        "    double* ws, double* wy, double* sy, double theta, CBLAS_INT col,\n"
        "    CBLAS_INT head, double* wa)\n{\n" + body + "}\n")
    (proposal.parent / "direction.c").write_text(patched + wrapper, newline="\n")
    records = {name: {"sha256": ex.digest(ex.WORK / "sources" / name / "lbfgsb.c")}
               for name in ex.VARIANTS}
    records["patch_changes"] = {
        "added": sum(s.startswith("+") and not s.startswith("+++") for s in ex.PATCH.read_text().splitlines()),
        "removed": sum(s.startswith("-") and not s.startswith("---") for s in ex.PATCH.read_text().splitlines())}
    assert records["patch_changes"] == {"added": 37, "removed": 2}
    return records


def build():
    compiler = ex.ENVIRONMENT / "bin/x86_64-conda-linux-gnu-cc"
    flags = [str(compiler), "-std=c17", "-O2", "-g", "-fvisibility=hidden", "-fPIC",
             "-march=nocona", "-mtune=haswell", "-ftree-vectorize", "-fstack-protector-strong",
             "-fno-plt", "-ffunction-sections", "-pipe", "-DNDEBUG", "-D_FORTIFY_SOURCE=2",
             "-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION", "-Wall", "-Wno-unused-but-set-variable",
             "-Wno-unused-function", "-Wno-conversion", "-Wno-misleading-indentation",
             f"-I{ex.np.get_include()}", f"-I{ex.ENVIRONMENT}/include/python3.13",
             f"-I{ex.SCIPY_SOURCE}/scipy/optimize/src",
             f"-I{ex.SCIPY_SOURCE}/scipy/_build_utils/src"]
    link = ["-shared", f"-L{ex.ENVIRONMENT}/lib", f"-Wl,-rpath,{ex.ENVIRONMENT}/lib",
            "-Wl,--disable-new-dtags", "-Wl,--as-needed", "-lopenblas", "-lm"]
    commands = []
    for variant in ex.VARIANTS:
        output = ex.WORK / "build" / variant
        output.mkdir(parents=True, exist_ok=True)
        source = ex.WORK / "sources" / variant
        modes = ("module", "direction") if variant == "proposal" else ("module",)
        for mode in modes:
            if mode == "module":
                inputs = [str(ex.SCIPY_SOURCE / "scipy/optimize/_lbfgsbmodule.c"), str(source / "lbfgsb.c")]
                target = output / ("_lbfgsb" + sysconfig.get_config_var("EXT_SUFFIX"))
            else:
                inputs, target = [str(source / "direction.c")], output / "direction.so"
            command = flags + inputs + link + ["-o", str(target)]
            result = subprocess.run(command, capture_output=True, text=True)
            (output / f"{mode}_compiler.log").write_text(result.stdout + result.stderr)
            result.check_returncode()
            commands.append(command)
        print(f"Built {variant}", flush=True)
    return commands


def main():
    sources = prepare()
    commands = build()
    info = {**ex.metadata(), "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "compiler": subprocess.check_output(
                [str(ex.ENVIRONMENT / "bin/x86_64-conda-linux-gnu-cc"), "--version"], text=True),
            "source_records": sources, "commands": commands,
            "binary_sha256": {name: ex.digest(ex.WORK / "build" / name /
                ("_lbfgsb" + sysconfig.get_config_var("EXT_SUFFIX"))) for name in ex.VARIANTS}}
    (ex.RESULTS / "build.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
