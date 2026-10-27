import subprocess
import sys


def test_orthology_and_evaluate_do_not_import_torch():
    code = "import sys, scgrip.orthology, scgrip.evaluate; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_public_api_resolves():
    import scgrip
    for name in scgrip.__all__:
        assert getattr(scgrip, name) is not None
