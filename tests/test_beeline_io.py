import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "beeline"))
from beeline_io import read_network, to_adjacency  # noqa: E402


def test_read_network_signs_and_conflicts(tmp_path):
    p = tmp_path / "net.csv"
    p.write_text("Gene1,Gene2,Type\nsox9,amh,+\nSOX9,AMH,+\nA,B,-\nA,B,+\nC,C,+\nC,D,-\n")
    net = read_network(p).set_index(["Gene1", "Gene2"]).sign.to_dict()
    assert net == {("SOX9", "AMH"): 1, ("A", "B"): 0, ("C", "D"): -1}


def test_trrust_raw_format(tmp_path):
    p = tmp_path / "trrust.tsv"
    p.write_text("TP53\tCDKN1A\tActivation\t123\nTP53\tBAX\tUnknown\t1\nMYC\tCDKN1A\tRepression\t2\n")
    net = read_network(p).set_index(["Gene1", "Gene2"]).sign.to_dict()
    assert net == {("MYC", "CDKN1A"): -1, ("TP53", "BAX"): 0, ("TP53", "CDKN1A"): 1}
    adj = to_adjacency(read_network(p))
    assert adj.loc["TP53", "BAX"] == 1 and adj.loc["MYC", "BAX"] == 0
