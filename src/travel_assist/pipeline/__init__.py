"""Retrieved chunks become a cited answer, inside a token budget.

Day 1's steps, built here as plain functions and composed into one deterministic
runnable in a later step. Day 2 takes the composition apart, not the steps —
`pipeline/steps.py::generate_answer` is called from both the LCEL chain and,
unmodified, from the agent's tool loop.
"""
