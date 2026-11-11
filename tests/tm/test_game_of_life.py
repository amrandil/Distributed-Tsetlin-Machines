"""Game of Life data, reference clause set and clause analysis."""

import numpy as np
import torch

from cakit.tm import HardTM
from experiments.relaxed_tm.game_of_life.clauses import analyze_bank, merged_rules, pack_bank, unpack_bank, wrong_rows
from experiments.relaxed_tm.game_of_life.data import (
    gol_step, gol_table, next_state, pattern_grid, patches, random_grid, row_index,
)
from experiments.relaxed_tm.game_of_life.reference import reference_cases, reference_tm


def test_table_counts():
    X, y = gol_table()
    assert X.shape == (512, 9) and len(np.unique(row_index(X))) == 512
    assert y.sum() == 140  # 112 with exactly 3 neighbours + 28 alive with exactly 2
    # row 0b000_010_000 = lone live cell dies; 0b111_000_000 (3 neighbours, centre dead) is born
    assert y[0b000010000] == 0 and y[0b111000000] == 1


def test_reference_is_exact_and_190_clauses():
    ref = reference_tm()
    X, y = gol_table()
    assert ref.n_clauses == 190
    assert int((ref.polarity > 0).sum()) == 84
    assert (ref.predict(torch.as_tensor(X)).numpy() == y).all()
    assert [c[2:] for c in reference_cases()] == [(56, 8), (28, 9), (70, 4), (8, 7), (28, 9)]


def test_reference_is_pure_and_duplicate_free():
    st = analyze_bank(reference_tm(), target=1)["total"]
    assert st["active"] == 190 and st["distinct"] == 190
    assert st["pure_fraction"] == 1.0 and st["distinct_in_reference"] == 190


def test_patches_match_table_rule():
    grid = random_grid(20, seed=3)
    p = patches(grid)
    assert p.shape == (400, 9)
    # the centre bit of each patch is the cell itself
    assert (p[:, 4] == grid.reshape(-1)).all()
    # the NW bit of cell (r, c) is grid[r-1, c-1]
    assert p[5 * 20 + 7, 0] == grid[4, 6]
    assert p[0, 0] == grid[-1, -1]  # wraps around


def test_true_step_on_known_patterns():
    block = pattern_grid("block")
    assert (gol_step(block) == block).all()
    blinker = pattern_grid("blinker")
    assert not (gol_step(blinker) == blinker).all()
    assert (gol_step(gol_step(blinker)) == blinker).all()
    glider = pattern_grid("glider")
    g = glider
    for _ in range(4):
        g = gol_step(g)
    assert (g == np.roll(glider, (1, 1), axis=(0, 1))).all()


def test_step_equals_rule_on_patches():
    grid = random_grid(16, seed=0)
    assert (gol_step(grid).reshape(-1) == next_state(patches(grid))).all()


def test_analysis_counts_duplicates_and_impure():
    ref = reference_tm()
    # two copies of the first positive clause, one impure positive clause (fires on everything alive-centred)
    impure = torch.zeros(1, 18, dtype=torch.bool); impure[0, 4] = True
    include = torch.cat([ref.include[:1], ref.include[:1], impure, ref.include[84:85]])
    bank = HardTM(include=include, polarity=torch.tensor([1.0, 1.0, 1.0, -1.0]))
    st = analyze_bank(bank, target=1)
    assert st["pos"]["active"] == 3 and st["pos"]["distinct"] == 2
    assert st["pos"]["pure"] == 2 and st["neg"]["pure"] == 1
    assert any("(×2)" in line for line in merged_rules(bank))


def test_pack_roundtrip_and_wrong_rows():
    ref = reference_tm()
    back = unpack_bank(pack_bank(ref))
    assert torch.equal(back.include, ref.include) and torch.equal(back.polarity, ref.polarity)
    assert wrong_rows(ref.predict) == []
    always_dead = lambda x: torch.zeros(len(x), dtype=torch.long)
    assert len(wrong_rows(always_dead)) == 140


def test_multiclass_analysis_counts_shared_clauses():
    from experiments.relaxed_tm.game_of_life.clauses import analyze_multiclass
    ref = reference_tm()
    alive = ref                                                           # + votes alive
    dead = HardTM(include=ref.include.clone(), polarity=-ref.polarity)    # + votes dead: same clauses, same votes
    st = analyze_multiclass([dead, alive])["total"]
    assert st["active"] == 380 and st["distinct"] == 190 and st["shared_across"] == 190
    assert st["pure_fraction"] == 1.0 and st["distinct_in_reference"] == 190


def test_granmo_single_tm_conversion_matches_his_predictions():
    import pytest
    pytest.importorskip("Cython")
    from experiments.relaxed_tm.classification.baselines import Granmo2018Config, run_granmo2018_single_tm
    X, y = gol_table()
    r = run_granmo2018_single_tm(X, y, X, y, Granmo2018Config(clauses_total=40, T=10, s=3.0, epochs=3, eval_every=1), seed=0)
    assert r["hard_matches_original"]
    assert len(r["test_curve"]) == 3
