"""Tests for edgetutor.core.jetson — hardware detection and model recommendation."""

from edgetutor.core.jetson import (
    JETSON_PROFILES,
    MODEL_TIERS,
    JetsonProfile,
)


class TestJetsonProfiles:
    """Test that all profiles are valid and complete."""

    def test_all_profiles_have_required_fields(self):
        """Every profile should have all required fields."""
        for key, profile in JETSON_PROFILES.items():
            assert isinstance(profile, JetsonProfile), f"{key} is not a JetsonProfile"
            assert profile.name, f"{key} missing name"
            assert profile.ram_gb > 0, f"{key} has invalid RAM"
            assert profile.recommended_llm, f"{key} missing recommended LLM"
            assert profile.recommended_context_size > 0, f"{key} has invalid context size"

    def test_known_profiles_exist(self):
        """Key Jetson models should have profiles."""
        expected = ["orin_nano_8gb", "orin_nano_4gb", "orin_nx_16gb", "cpu_only"]
        for key in expected:
            assert key in JETSON_PROFILES, f"Missing profile: {key}"

    def test_orin_nano_8gb_profile(self):
        """Orin Nano 8GB should recommend Phi-3 Mini."""
        profile = JETSON_PROFILES["orin_nano_8gb"]
        assert profile.ram_gb == 8.0
        assert "Phi-3" in profile.recommended_llm or "phi" in profile.recommended_llm.lower()
        assert profile.recommended_gpu_layers > 0

    def test_cpu_only_has_zero_gpu_layers(self):
        """CPU-only profile should have 0 GPU layers."""
        profile = JETSON_PROFILES["cpu_only"]
        assert profile.recommended_gpu_layers == 0

    def test_4gb_boards_use_tiny_models(self):
        """4GB boards should recommend TinyLlama or similar small model."""
        for key in ["orin_nano_4gb", "nano_4gb"]:
            profile = JETSON_PROFILES[key]
            assert profile.recommended_llm_size_gb <= 1.5, (
                f"{key} recommends too large a model for {profile.ram_gb}GB RAM"
            )


class TestModelTiers:
    """Test model tier configuration."""

    def test_tiers_are_ordered_by_size(self):
        """Model tiers should be ordered from largest to smallest."""
        for i in range(len(MODEL_TIERS) - 1):
            assert MODEL_TIERS[i]["size_gb"] >= MODEL_TIERS[i + 1]["size_gb"], (
                f"Model tiers not ordered: {MODEL_TIERS[i]['name']} > {MODEL_TIERS[i + 1]['name']}"
            )

    def test_all_tiers_have_required_fields(self):
        """Every tier should have name, filename, url, size."""
        for tier in MODEL_TIERS:
            assert "name" in tier
            assert "filename" in tier
            assert "url" in tier
            assert "size_gb" in tier
            assert tier["size_gb"] > 0

    def test_min_ram_is_reasonable(self):
        """Minimum RAM should be less than model size * 3 (generous overhead)."""
        for tier in MODEL_TIERS:
            assert tier["min_ram_gb"] <= tier["size_gb"] * 3
