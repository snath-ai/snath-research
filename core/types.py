"""
Shared type definitions for Snath Research.
"""
from enum import Enum


class RouteDecision(str, Enum):
    COMMIT_TRAJECTORY  = "COMMIT_TRAJECTORY"   # streams agree — proceed
    TRIGGER_REPLAN     = "TRIGGER_REPLAN"       # recoverable contradiction — investigate
    STRUCTURAL_IMPASSE = "STRUCTURAL_IMPASSE"   # irreconcilable — halt
    DEFER              = "DEFER"                # one stream uncertain — lean on confident arm
