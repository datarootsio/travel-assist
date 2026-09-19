"""MLflow tracing, local by default.

A directory (`./mlruns`) rather than a service, so there is nothing to provision
and nothing that can be unreachable on a teaching morning. `make mlflow-ui`
renders it.

Prompt management is deliberately out of scope.
Prompts live in Python next to the code that uses them; a registry is worth one
paragraph in the guide and no code in this repo.
"""
