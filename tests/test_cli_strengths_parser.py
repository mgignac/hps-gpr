from hps_gpr.cli import _parse_mass_tokens, _parse_strength_tokens


def test_parse_strength_tokens_supports_sigma_prefix():
    vals = _parse_strength_tokens("s1,s2,3,s5")
    assert vals == [1.0, 2.0, 3.0, 5.0]


def test_parse_strength_tokens_none_returns_empty():
    assert _parse_strength_tokens(None) == []


def test_parse_mass_tokens_supports_comma_separated_values():
    vals = _parse_mass_tokens("0.040,0.080,0.120")
    assert vals == [0.04, 0.08, 0.12]


def test_parse_mass_tokens_none_returns_empty():
    assert _parse_mass_tokens(None) == []
