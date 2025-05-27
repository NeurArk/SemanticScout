from core.utils.vector_utils import cosine_similarity


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    sim = cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert 0.0 <= sim <= 0.1
