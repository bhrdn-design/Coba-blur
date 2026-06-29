"""Tes cepat logika is_peace() pakai landmark sintetis (tanpa webcam)."""

from types import SimpleNamespace

from peace_blur import is_peace


def make_hand(index_up, middle_up, ring_up, pinky_up, spread=0.2):
    """Bikin 21 landmark dummy. Tangan tegak (jari naik = y lebih kecil).
    wrist y=0.9, mid_mcp y=0.6 -> hand_size ~0.3."""
    pts = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    pts[0] = SimpleNamespace(x=0.5, y=0.9, z=0.0)   # wrist
    pts[9] = SimpleNamespace(x=0.5, y=0.6, z=0.0)   # middle_mcp

    def set_finger(pip_i, tip_i, up, tip_x):
        pts[pip_i] = SimpleNamespace(x=tip_x, y=0.5, z=0.0)
        pts[tip_i] = SimpleNamespace(x=tip_x, y=(0.2 if up else 0.6), z=0.0)

    set_finger(6, 8, index_up, 0.5 - spread / 2)    # index
    set_finger(10, 12, middle_up, 0.5 + spread / 2)  # middle
    set_finger(14, 16, ring_up, 0.6)                 # ring
    set_finger(18, 20, pinky_up, 0.7)                # pinky
    return pts


def test_peace_detected():
    assert is_peace(make_hand(True, True, False, False)) is True


def test_fist_not_peace():
    assert is_peace(make_hand(False, False, False, False)) is False


def test_open_palm_not_peace():
    assert is_peace(make_hand(True, True, True, True)) is False


def test_fingers_too_close_not_peace():
    # telunjuk+tengah naik tapi rapat banget -> bukan bentuk V
    assert is_peace(make_hand(True, True, False, False, spread=0.02)) is False


if __name__ == "__main__":
    tests = [
        test_peace_detected,
        test_fist_not_peace,
        test_open_palm_not_peace,
        test_fingers_too_close_not_peace,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print("\nSemua tes lolos ✅")
