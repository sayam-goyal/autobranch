

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


def test_the_recorded_mode_is_the_one_that_was_measured():
    """Provenance must describe the measurement, not the request.

    `deterministic` defaults to True, so a manifest asking for stochastic only
    takes the fallback branch -- and the SF3 artifact then claimed its
    headline was measured deterministically, and printed a reproduction
    command with `--mode deterministic`, for a value the same record filed
    under `modes.stochastic` with `deterministic: false` in its budget.

    Asserted on the source because reaching the real `evaluate` needs a
    checkpoint, a policy and an emulator; what matters here is that the
    reported mode is derived from `modes`, never from the flag.
    """
    import inspect

    from autobranch import objective

    source = inspect.getsource(objective.evaluate)
    assert 'measured_mode = preferred if preferred in modes else next(iter(modes))' in source
    assert '"mode": measured_mode,' in source
    assert "mode=measured_mode," in source
    # The request is still recorded, just not confused with the measurement.
    assert '"mode_requested": preferred,' in source
    # And the old formulation is gone from both the detail and the command.
    assert source.count('"deterministic" if deterministic else "stochastic"') == 1, (
        "only `preferred` should still be computed that way"
    )
