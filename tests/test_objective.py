

def test_the_evaluator_defaults_to_a_gpu_when_there_is_one():
    """Scoring on CPU a policy that was trained on a GPU is never the intent.

    Harmless for a two-layer MLP, fatal for an 8.77B vision-language policy: a
    forward per decision takes seconds on CPU, so a 24-match evaluation at
    ~1600 decisions a match would not finish inside any run's deadline. The
    symptom would have been an evaluation that simply never returned -- no
    error, no progress.
    """
    from autobranch import objective

    assert objective._default_device() in ("cuda", "cpu")
    # The signature's default is "pick one", not a hardcoded device.
    import inspect

    assert inspect.signature(objective.evaluate).parameters["device"].default == ""


def test_a_manifest_can_ask_for_one_evaluation_mode():
    """Both modes is the right default; it is also twice the cost.

    The gap between them is diagnostic, so it stays on by default. But each
    mode is a full pass over the match budget, and where one decision costs a
    quarter second of emulator time that is a 24-minute screen versus a
    48-minute one. An unknown mode is refused rather than ignored, because
    silently measuring both after being asked for one is the expensive
    surprise.
    """
    import inspect

    from autobranch import objective

    source = inspect.getsource(objective.evaluate)
    assert 'merged_budget.get("modes")' in source
    assert 'if mode_name not in wanted' in source
    # mode_gap is only meaningful when both were measured.
    assert "if len(modes) == 2 else None" in source
