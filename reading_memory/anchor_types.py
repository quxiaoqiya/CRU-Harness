"""Open-world reading ontology for evidence-aware document understanding."""
from __future__ import annotations
from enum import Enum

class AnchorType(str, Enum):
    QUESTION = "question"; OBJECTIVE = "objective"; HYPOTHESIS = "hypothesis"; CONCLUSION = "conclusion"
    ENTITY = "entity"; MATERIAL = "material"; SAMPLE = "sample"; DEVICE = "device"; METHOD = "method"
    EVENT = "event"; SYNTHESIS = "synthesis"; PREPARATION = "preparation"; TEST = "test"; CHARACTERIZATION = "characterization"
    CONDITION = "condition"; PARAMETER = "parameter"; VALUE = "value"; TREND = "trend"; COMPARISON = "comparison"
    FIGURE = "figure"; TABLE = "table"; REFERENCE = "reference"; NEGATION = "negation"; UNCERTAINTY = "uncertainty"
    LIMITATION = "limitation"; CROSS_REFERENCE = "cross_reference"; ALIAS = "alias"; MISSING = "missing"; CONFLICT = "conflict"; TASK = "task"
