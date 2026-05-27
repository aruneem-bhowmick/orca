"""Prompt template for explaining a specific transfer learning recommendation."""

TRANSFER_EXPLANATION_TEMPLATE = """\
You are an expert in transfer learning. A user wants to transfer knowledge \
from a source task to a target task.

Source Task: {source_task_description}
Target Task: {target_task_description}
Transfer Score: {transfer_score:.2f}
Recommended Layers: {recommended_layers}

Explain in 2-3 sentences why this transfer should (or should not) work,
and what the user should expect in terms of performance improvement.
"""
