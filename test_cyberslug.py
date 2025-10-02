"""
test_cyberslug.py - Unit tests for CyberSlug MESA model
Run with: pytest test_cyberslug.py
"""
import pytest
import numpy as np
from model import CyberSlugModel
from agents import CyberslugAgent, PreyAgent


class TestCyberSlugModel:
    """Test suite for CyberSlugModel"""

    def test_model_initialization(self):
        """Test that model initializes correctly"""
        model = CyberSlugModel()

        assert model.width == 600
        assert model.height == 600
        assert model.ticks == 0
        assert model.cyberslug is not None
        assert len(model.schedule.agents) > 0

    def test_population_settings(self):
        """Test prey population settings"""
        hermi, flab, fauxflab = 10, 5, 3
        model = CyberSlugModel(
            hermi_population=hermi,
            flab_population=flab,
            fauxflab_population=fauxflab
        )

        # Count prey types
        prey_counts = {'hermi': 0, 'flab': 0, 'fauxflab': 0}
        for agent in model.schedule.agents:
            if isinstance(agent, PreyAgent):
                prey_counts[agent.prey_type] += 1

        assert prey_counts['hermi'] == hermi
        assert prey_counts['flab'] == flab
        assert prey_counts['fauxflab'] == fauxflab

    def test_model_step(self):
        """Test that model can step without errors"""
        model = CyberSlugModel()
        initial_ticks = model.ticks

        model.step()

        assert model.ticks == initial_ticks + 1

    def test_multiple_steps(self):
        """Test running multiple steps"""
        model = CyberSlugModel()
        steps = 10

        for _ in range(steps):
            model.step()

        assert model.ticks == steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])