from digits.train import evaluate_all, get_split, worst_confusions


def test_split_shapes_and_scaling():
    Xtr, Xte, ytr, yte = get_split()
    assert Xtr.shape[1] == 64 and len(Xtr) + len(Xte) == 1797
    assert 0.0 <= Xtr.min() and Xtr.max() <= 1.0


def test_all_models_above_95_percent():
    scores = evaluate_all()
    assert set(scores) == {"logreg", "svm_rbf", "mlp"}
    assert all(v > 0.95 for v in scores.values())


def test_reproducible():
    assert evaluate_all() == evaluate_all()


def test_confusions_are_off_diagonal():
    for count, t, p in worst_confusions():
        assert t != p and count > 0
