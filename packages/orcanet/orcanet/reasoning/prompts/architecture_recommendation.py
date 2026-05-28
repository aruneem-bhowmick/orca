"""Prompt template for recommending a model architecture for a target task."""

ARCHITECTURE_RECOMMENDATION_TEMPLATE = """\
Given a source task that performs best with {source_architecture} and a target task \
with characteristics {target_task_features}, recommend the most suitable architecture
with a brief rationale (2 sentences max).
"""
