from conftest import load_script

ce = load_script("scripts/repro/check_env.py")

LOCK = """
version = 1
requires-python = ">=3.12"

[[package]]
name = "torch"
version = "2.8.0"
dependencies = [
    { name = "nvidia-cublas-cu12", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
]

[[package]]
name = "nvidia-cublas-cu12"
version = "12.8.4.1"

[[package]]
name = "vllm"
version = "0.11.2"

[[package]]
name = "colorama"
version = "0.4.6"

[[package]]
name = "small-models-clinical-decision"
version = "0.1.0"
source = { editable = "." }
dependencies = [
    { name = "torch" },
    { name = "vllm", marker = "sys_platform == 'linux'" },
]

[package.dev-dependencies]
dev = [
    { name = "colorama", marker = "sys_platform == 'win32'" },
]
"""

MAC = {"sys_platform": "darwin", "platform_machine": "arm64", "python_version": "3.13", "python_full_version": "3.13.1",
       "os_name": "posix", "platform_system": "Darwin", "platform_release": "", "platform_version": "", "implementation_name": "cpython",
       "implementation_version": "3.13.1", "platform_python_implementation": "CPython"}
LINUX = {**MAC, "sys_platform": "linux", "platform_machine": "x86_64", "platform_system": "Linux"}


def test_parse_lock_skips_editable_root():
    locked = ce.parse_lock(LOCK, env=LINUX)
    assert locked == {"torch": "2.8.0", "vllm": "0.11.2", "nvidia-cublas-cu12": "12.8.4.1"}


def test_parse_lock_honours_platform_markers():
    """CUDA wheels and Windows-only packages are locked for other platforms; on a Mac they are not drift."""
    locked = ce.parse_lock(LOCK, env=MAC)
    assert locked == {"torch": "2.8.0"}


def test_parse_lock_defaults_to_the_running_platform():
    locked = ce.parse_lock(LOCK)
    assert ("nvidia-cublas-cu12" in locked) == (ce.platform.system() == "Linux" and ce.platform.machine() == "x86_64")


def test_compare_reports_drift_and_missing():
    locked = {"torch": "2.8.0", "vllm": "0.11.2", "numpy": "2.1.0"}
    installed = {"torch": "2.8.0", "vllm": "0.11.3"}
    drift = ce.compare(locked, installed)
    assert sorted((d["package"], d["problem"]) for d in drift) == [("numpy", "missing"), ("vllm", "version")]


def test_compare_normalises_names():
    assert ce.compare({"Flash_Linear-Attention": "0.3"}, {"flash-linear-attention": "0.3"}) == []


def test_check_pins_flags_wrong_kev_commit():
    pins = {"kev_commit": "569ea44"}
    assert ce.check_pins(pins, {"kev_commit": "569ea44abcdef"}) == []
    problems = ce.check_pins(pins, {"kev_commit": "1234567"})
    assert problems and problems[0]["pin"] == "kev_commit"
    assert ce.check_pins(pins, {})[0]["problem"] == "unknown"


def test_check_env_runs_in_the_project_env_not_an_isolated_script_env(repo_root):
    """A PEP 723 header makes `uv run` build an isolated env, so every locked package reads as missing."""
    assert "# /// script" not in (repo_root / "scripts/repro/check_env.py").read_text()


FORKED = """
version = 1
requires-python = ">=3.12"

[[package]]
name = "numpy"
version = "1.26.4"
resolution-markers = ["python_full_version < '3.13'"]

[[package]]
name = "numpy"
version = "2.3.0"
resolution-markers = ["python_full_version >= '3.13'"]

[[package]]
name = "small-models-clinical-decision"
version = "0.1.0"
source = { virtual = "." }
dependencies = [
    { name = "numpy", version = "1.26.4", marker = "python_full_version < '3.13'" },
    { name = "numpy", version = "2.3.0", marker = "python_full_version >= '3.13'" },
]

[package.dev-dependencies]
dev = [
    { name = "colorama" },
]
docs = [
    { name = "mkdocs" },
]

[[package]]
name = "colorama"
version = "0.4.6"

[[package]]
name = "mkdocs"
version = "1.6.0"
"""


def test_parse_lock_follows_the_edge_version_in_a_forked_lock():
    """A lock forked by Python version lists a package twice; the edge's `version` picks the entry."""
    py312 = {**MAC, "python_version": "3.12", "python_full_version": "3.12.4"}
    assert ce.parse_lock(FORKED, env=py312)["numpy"] == "1.26.4"
    assert ce.parse_lock(FORKED, env=MAC)["numpy"] == "2.3.0"


def test_parse_lock_walks_every_dependency_group():
    locked = ce.parse_lock(FORKED, env=MAC)
    assert locked["colorama"] == "0.4.6" and locked["mkdocs"] == "1.6.0"
