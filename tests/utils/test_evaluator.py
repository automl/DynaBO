import pytest
from ConfigSpace import Float

from dynabo.utils.evaluator import BBOBEvaluator


def test_setup_bbob_evaluator():
    evaluator = BBOBEvaluator(1, 1, 5)
    assert evaluator.scenario == 1
    assert evaluator.dataset == 1
    assert evaluator.dimension == 5


@pytest.fixture
def bbob_evaluator():
    return BBOBEvaluator(1, 1, 5)


def test_get_configuration_space(bbob_evaluator: BBOBEvaluator):
    configuration_space = bbob_evaluator.get_configuration_space()
    assert configuration_space is not None
    assert len(configuration_space.values()) == 5
    assert list(configuration_space.values()) == [
        Float(name="x0", bounds=[-5, 5]),
        Float(name="x1", bounds=[-5, 5]),
        Float(name="x2", bounds=[-5, 5]),
        Float(name="x3", bounds=[-5, 5]),
        Float(name="x4", bounds=[-5, 5]),
    ]


def test_evalaute_random_configuration(bbob_evaluator: BBOBEvaluator):
    configspace = bbob_evaluator.get_configuration_space()
    configspace.seed(1)
    config = configspace.sample_configuration()
    cost, runtime = bbob_evaluator.train(config)
    assert isinstance(cost, float)
    assert isinstance(runtime, float)
    assert cost == 121.58965959685241
    assert runtime == 0
