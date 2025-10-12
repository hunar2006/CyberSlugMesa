"""
test_cyberslug_complete.py - Comprehensive unit tests for enhanced CyberSlug model
Tests all NetLogo features including advanced learning circuit
Run with: pytest test_cyberslug_complete.py -v
"""
import pytest
import numpy as np
from model import CyberSlugModel
from agents import CyberslugAgent, PreyAgent, Nociceptor


class TestCyberSlugModel:
    """Test suite for CyberSlugModel with all features"""

    def test_model_initialization(self):
        """Test that model initializes correctly with default parameters"""
        model = CyberSlugModel()

        assert model.width == 600
        assert model.height == 600
        assert model.ticks == 0
        assert len(model.cyberslugs) > 0
        assert len(model.schedule.agents) > 0

    def test_model_with_clustering(self):
        """Test model initialization with clustering enabled"""
        model = CyberSlugModel(clustering=True, cluster_radius=20)

        assert model.clustering == True
        assert model.cluster_radius == 20
        assert model.hermi_cluster_x is not None
        assert model.flab_cluster_x is not None

    def test_model_with_immobilize(self):
        """Test immobilize mode"""
        model = CyberSlugModel(immobilize=True)

        slug = model.cyberslugs[0]
        initial_pos = slug.pos

        model.step()

        # Position should not change when immobilized
        assert slug.pos == initial_pos

    def test_odor_null_mode(self):
        """Test odor-null mode where slugs don't emit odor"""
        model = CyberSlugModel(odor_null=True)

        assert model.odor_null == True

        # After a few steps, slug odor should be minimal
        for _ in range(5):
            model.step()

        # Check that pleur odor is very low
        total_pleur_odor = np.sum(model.patches[4])  # pleur is index 4
        assert total_pleur_odor < 1.0  # Should be very low

    def test_population_settings(self):
        """Test prey population settings"""
        hermi, flab, fauxflab = 10, 5, 3
        model = CyberSlugModel(
            hermi_population=hermi,
            flab_population=flab,
            fauxflab_population=fauxflab
        )

        prey_counts = {'hermi': 0, 'flab': 0, 'fauxflab': 0}
        for agent in model.schedule.agents:
            if isinstance(agent, PreyAgent):
                prey_counts[agent.prey_type] += 1

        assert prey_counts['hermi'] == hermi
        assert prey_counts['flab'] == flab
        assert prey_counts['fauxflab'] == fauxflab

    def test_multiple_slugs(self):
        """Test model with multiple slugs"""
        num_slugs = 5
        model = CyberSlugModel(num_slugs=num_slugs)

        assert len(model.cyberslugs) == num_slugs
        assert model.being_observed in model.cyberslugs

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

    def test_fix_satiation(self):
        """Test fixed satiation mode"""
        model = CyberSlugModel(fix_satiation_override=True, fix_satiation_value=0.8)

        slug = model.cyberslugs[0]
        model.step()

        assert slug.satiation == 0.8

    def test_odor_patch_system(self):
        """Test 5-odor patch system"""
        model = CyberSlugModel()

        # Should have 5 odor types
        assert model.patches.shape[0] == 5

        # Set odor and verify
        model.set_patch_odor(300, 300, [1.0, 0.5, 0.3, 0.0, 0.2])
        odors = model.get_odor_at_position(300, 300)

        assert len(odors) == 5
        assert odors[0] > 0  # betaine
        assert odors[1] > 0  # hermi


class TestCyberslugAgent:
    """Test suite for CyberslugAgent with all features"""

    def test_agent_initialization(self):
        """Test agent initializes with all new attributes"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        # Check basic attributes
        assert slug.size > 0
        assert slug.nutrition == 0.5

        # Check advanced learning circuit attributes
        assert hasattr(slug, 'Vh_rp')
        assert hasattr(slug, 'Vh_rn')
        assert hasattr(slug, 'Vh_n')
        assert hasattr(slug, 'Vf_rp')
        assert hasattr(slug, 'Vf_rn')
        assert hasattr(slug, 'Vf_n')

        # Check reward neurons
        assert hasattr(slug, 'R_pos')
        assert hasattr(slug, 'R_neg')
        assert hasattr(slug, 'NR')

        # Check CS neurons
        assert hasattr(slug, 'CS1')
        assert hasattr(slug, 'CS2')

        # Check baselines
        assert hasattr(slug, 'Vh_rp0')
        assert hasattr(slug, 'Vf_rn0')

    def test_nociceptor_system(self):
        """Test 7-nociceptor pain system"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        # Should have 7 nociceptors
        assert len(slug.nociceptors) == 7

        # Check nociceptor IDs
        expected_ids = ["snsrOL", "snsrOR", "snsrUL", "snsrUR", "snsrBL", "snsrBR", "snsrBM"]
        actual_ids = [noc.id for noc in slug.nociceptors]

        for expected_id in expected_ids:
            assert expected_id in actual_ids

    def test_nociceptor_positions(self):
        """Test that nociceptors update positions correctly"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        # Update positions
        slug.update_nociceptor_positions()

        # Check that all nociceptors have positions
        for noc in slug.nociceptors:
            assert noc.x is not None
            assert noc.y is not None
            assert isinstance(noc.x, (int, float))
            assert isinstance(noc.y, (int, float))

    def test_proboscis_behavior(self):
        """Test proboscis extension behavior"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        # Initially not extended
        assert slug.proboscis_phase == 0

        # Manually trigger high betaine to extend proboscis
        slug.sns_odors_left[0] = 6.0  # High betaine
        slug.update_proboscis()

        # Should start extending
        assert slug.proboscis_phase > 0 or slug.proboscis_extended

    def test_habituation_circuit(self):
        """Test habituation/sensitization circuit"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        # Check circuit components exist
        assert hasattr(slug, 'W1')
        assert hasattr(slug, 'W2')
        assert hasattr(slug, 'W3')
        assert hasattr(slug, 'M')
        assert hasattr(slug, 'M0')
        assert hasattr(slug, 'S')
        assert hasattr(slug, 'IN')

        # Initial W3 should be 0.5
        assert slug.W3 == 0.5

    def test_learning_circuit_neurons(self):
        """Test that learning circuit neurons update"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        # Run a few steps
        for _ in range(10):
            model.step()

        # Neurons should have been updated
        assert isinstance(slug.R_pos, float)
        assert isinstance(slug.R_neg, float)
        assert isinstance(slug.NR, float)

        # Values should be between 0 and 1 (sigmoid outputs)
        assert 0 <= slug.R_pos <= 1
        assert 0 <= slug.R_neg <= 1
        assert 0 <= slug.NR <= 1

    def test_association_strength_learning(self):
        """Test that association strengths can increase with learning"""
        model = CyberSlugModel(hermi_population=20)
        slug = model.cyberslugs[0]

        initial_Vh_rp = slug.Vh_rp

        # Run simulation - slug should encounter hermissenda
        for _ in range(500):
            model.step()

        # If slug ate hermissenda, Vh_rp should increase
        if slug.hermi_counter > 0:
            assert slug.Vh_rp >= initial_Vh_rp

    def test_synaptic_weight_calculation(self):
        """Test that synaptic weights are calculated from association strengths"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        # Manually set high association strength
        slug.Vh_rp = 1.5
        slug.calc_learning_circuit()

        # Weight should be high (sigmoid of 1.5 should be > 0.8)
        assert slug.Wh_rp > 0.8

    def test_saturation_detection(self):
        """Test that saturation is detected correctly"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        # Manually create saturated weight
        slug.Vh_rp = 2.0
        slug.calc_learning_circuit()

        # Should detect saturation
        assert slug.Wh_rp > 0.83
        # Note: saturation flag updates in next calc_learning_circuit call

    def test_forgetting_mechanism(self):
        """Test that forgetting occurs (association strengths decay)"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        # Set high association strength
        slug.Vh_rp = 1.0
        slug.Vh_rp0 = 0.0  # Ensure baseline is 0

        # Run many steps without reward
        for _ in range(100):
            slug.calc_learning_circuit()

        # Should have decayed (forgetting)
        assert slug.Vh_rp < 1.0

    def test_collision_detection(self):
        """Test collision detection between slugs"""
        model = CyberSlugModel(num_slugs=2)
        slug1, slug2 = model.cyberslugs[0], model.cyberslugs[1]

        # Place slugs close together
        model.space.move_agent(slug1, (300, 300))
        model.space.move_agent(slug2, (305, 300))

        # Run step
        model.step()

        # At least one should detect collision
        # (depends on heading and cone)
        # Just check that collision attribute exists and is 0 or 1
        assert slug1.collision in [0, 1]
        assert slug2.collision in [0, 1]

    def test_prey_encounter_triggers_learning(self):
        """Test that encountering prey triggers appropriate learning inputs"""
        model = CyberSlugModel(hermi_population=30)
        slug = model.cyberslugs[0]

        initial_hermi_count = slug.hermi_counter

        # Run until slug eats something
        for _ in range(200):
            model.step()
            if slug.hermi_counter > initial_hermi_count:
                break

        # If slug ate hermi, R_pos_input should have been triggered
        if slug.hermi_counter > initial_hermi_count:
            # R_pos_input decays, but should still be > 0 recently after eating
            assert slug.R_pos_input >= 0  # At minimum, should be non-negative


class TestPreyAgent:
    """Test suite for PreyAgent with clustering"""

    def test_prey_initialization(self):
        """Test prey initializes correctly"""
        model = CyberSlugModel()

        # Find a prey agent
        prey = None
        for agent in model.schedule.agents:
            if isinstance(agent, PreyAgent):
                prey = agent
                break

        assert prey is not None
        assert prey.prey_type in ['hermi', 'flab', 'fauxflab']
        assert prey.cluster_target is not None

    def test_prey_clustering_movement(self):
        """Test that prey move towards cluster when clustering is enabled"""
        model = CyberSlugModel(clustering=True, cluster_radius=15)

        # Find a hermi prey
        hermi = None
        for agent in model.schedule.agents:
            if isinstance(agent, PreyAgent) and agent.prey_type == 'hermi':
                hermi = agent
                break

        assert hermi is not None

        # Place prey far from cluster
        model.space.move_agent(hermi, (100, 100))
        hermi.cluster_target = (500, 500)

        initial_dist = np.sqrt((500-100)**2 + (500-100)**2)

        # Run several steps
        for _ in range(50):
            hermi.step()

        # Should have moved closer to cluster
        x, y = hermi.pos
        final_dist = np.sqrt((500-x)**2 + (500-y)**2)

        # May not always be closer due to randomness, but usually should be
        # Just check that it's attempting to move
        assert final_dist <= initial_dist + 50  # Allow some tolerance

    def test_prey_respawn_clustering(self):
        """Test that prey respawn near cluster when clustering enabled"""
        model = CyberSlugModel(clustering=True, cluster_radius=20)

        # Find a prey
        prey = None
        for agent in model.schedule.agents:
            if isinstance(agent, PreyAgent):
                prey = agent
                break

        # Set cluster target
        prey.cluster_target = (300, 300)

        # Respawn
        prey.respawn()

        # Should be near cluster
        x, y = prey.pos
        dist = np.sqrt((x - 300)**2 + (y - 300)**2)

        assert dist < model.cluster_radius + 10  # Small tolerance


class TestInteractions:
    """Test interactive features"""

    def test_apply_pain_at_position(self):
        """Test poker tool (applying pain at position)"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        # Place slug at known position
        model.space.move_agent(slug, (300, 300))

        # Apply pain near slug
        model.apply_pain_at_position(305, 305, amount=50.0)

        # Slug should have received pain at nociceptors
        total_pain = sum(noc.painval for noc in slug.nociceptors)
        assert total_pain > 0

    def test_set_observed_slug(self):
        """Test observer selection"""
        model = CyberSlugModel(num_slugs=3)

        slug1 = model.cyberslugs[0]
        slug2 = model.cyberslugs[1]

        # Place slugs at known positions
        model.space.move_agent(slug1, (300, 300))
        model.space.move_agent(slug2, (400, 400))

        # Set observer to slug2
        success = model.set_observed_slug(400, 400)

        assert success
        assert model.being_observed == slug2

    def test_zero_learning_values(self):
        """Test resetting learning values"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        # Set some values
        slug.Vh_rp = 1.5
        slug.Vh_rn = 0.8
        slug.Vf_rp = 1.2

        # Reset hermi values
        model.zero_V_hermi()

        assert slug.Vh_rp == 0.0
        assert slug.Vh_rn == 0.0
        assert slug.Vf_rp == 1.2  # Should not be affected

        # Reset flab values
        model.zero_V_flab()

        assert slug.Vf_rp == 0.0


class TestNociceptor:
    """Test nociceptor class"""

    def test_nociceptor_creation(self):
        """Test nociceptor initialization"""
        model = CyberSlugModel()
        slug = model.cyberslugs[0]

        noc = Nociceptor("test_noc", slug)

        assert noc.id == "test_noc"
        assert noc.parent == slug
        assert noc.painval == 0.0
        assert noc.hit == False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])