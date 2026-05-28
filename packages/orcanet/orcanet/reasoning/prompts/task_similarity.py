"""Prompt template for assessing task similarity for transfer learning."""

TASK_SIMILARITY_TEMPLATE = """\
Compare these two ML tasks and assess their similarity for transfer learning purposes.
Rate similarity 0-1 and explain the key factors.

Task A: {task_a_description}
Task B: {task_b_description}
"""
