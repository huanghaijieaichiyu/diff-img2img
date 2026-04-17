from core.engine import DiffusionEngine


def test_ordered_weight_candidates_prefers_best_model_files_before_parent_fallback():
    engine = DiffusionEngine.__new__(DiffusionEngine)
    candidates = engine._ordered_weight_candidates(
        "/tmp/run/best_model",
        "decom_model_best.pth",
        "decom_model.pth",
    )

    assert candidates[:4] == [
        "/tmp/run/best_model/unet_final/decom_model_best.pth",
        "/tmp/run/best_model/decom_model_best.pth",
        "/tmp/run/decom_model_best.pth",
        "/tmp/run/best_model/unet_final/decom_model.pth",
    ]
    assert candidates[4:] == [
        "/tmp/run/best_model/decom_model.pth",
        "/tmp/run/decom_model.pth",
    ]
