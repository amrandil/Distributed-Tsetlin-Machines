"""
Feedback sub-package.

Provides abstract base classes and concrete implementations for feedback functions.
"""

from .base import FeedbackFunction
from .neighbourhood_agreement import NeighbourhoodAgreementFeedback
from .minority_disagreement import MinorityDisagreementFeedback
from .global_target import GlobalTargetFeedback

__all__ = [
    'FeedbackFunction',
    'NeighbourhoodAgreementFeedback',
    'MinorityDisagreementFeedback',
    'GlobalTargetFeedback',
]
