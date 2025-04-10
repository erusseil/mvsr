from pathlib import Path

from mvsr import MvSR


TEST_ROOT = Path(__file__).parent
DATASETS_ROOT = TEST_ROOT.parent.parent / "datasets"


def test_linear():
    mv = MvSR(DATASETS_ROOT / "linear/", max_length=10, pop_size=100, generations=100, n_params=3)
    mv.run()
    result_table = mv.raw_results
    mv.plot_all_fits()
