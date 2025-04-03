from unittest.mock import MagicMock

from gas.models import BaseModel


def test_base_model():
    # Mock base model
    GENERATED_OUTPUT = "This is a test output."
    mock_base_model = MagicMock(spec=BaseModel)
    mock_base_model.generate.return_value = GENERATED_OUTPUT

    question = "The dog chased the cat up the tree, who ran up the tree?"
    actual_output = mock_base_model.generate(question)
    assert len(actual_output) > 0
