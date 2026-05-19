"""
Utility functions for the FWA risk scoring pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def setup_directories():
    """Create all required output and data directories."""
    dirs = [
        config.DATA_RAW,
        config.DATA_PROCESSED,
        config.DATA_DOCUMENTS,
        config.OUTPUTS_FIGURES,
        config.OUTPUTS_MODELS,
        config.OUTPUTS_REPORTS,
        config.OUTPUTS_REVIEWS,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"  Ensured {len(dirs)} directories exist.")


def load_config():
    """Return a dict of all config values."""
    return {
        "BASE_DIR":             config.BASE_DIR,
        "DATA_RAW":             config.DATA_RAW,
        "DATA_PROCESSED":       config.DATA_PROCESSED,
        "DATA_DOCUMENTS":       config.DATA_DOCUMENTS,
        "OUTPUTS_FIGURES":      config.OUTPUTS_FIGURES,
        "OUTPUTS_MODELS":       config.OUTPUTS_MODELS,
        "OUTPUTS_REPORTS":      config.OUTPUTS_REPORTS,
        "OUTPUTS_REVIEWS":      config.OUTPUTS_REVIEWS,
        "N_CLAIMS":             config.N_CLAIMS,
        "RANDOM_SEED":          config.RANDOM_SEED,
        "FRAUD_BASE_RATE":      config.FRAUD_BASE_RATE,
        "HIGH_RISK_THRESHOLD":  config.HIGH_RISK_THRESHOLD,
    }


def format_currency(amount):
    """Format a numeric amount as a USD currency string."""
    try:
        return f"${float(amount):,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def get_risk_level(score):
    """
    Map a numeric risk score (0–1) to a human-readable risk level.

    Returns: 'Low', 'Medium', or 'High'
    """
    try:
        score = float(score)
    except (ValueError, TypeError):
        return "Unknown"

    if score >= config.HIGH_RISK_THRESHOLD:
        return "High"
    elif score >= 0.3:
        return "Medium"
    return "Low"
