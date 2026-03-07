# ARC-AGI-2 Archived Tasks

## Visual Example

|                                  Puzzle                                   |                                  Solution                                   |
| :-----------------------------------------------------------------------: | :-------------------------------------------------------------------------: |
| <img src="../../assets/examples/arcagi2/arcagi2_puzzle.png" width="350"/> | <img src="../../assets/examples/arcagi2/arcagi2_solution.png" width="350"/> |

## Task Description

`ARC-AGI-2` contains abstract reasoning tasks that require few-shot learning from a small number of solved demonstrations.

Each sample presents several input-output examples together with one held-out test input. The model must infer the hidden transformation rule and generate the correct output grid for the test case.

## What This Task Tests

This task primarily evaluates whether a model can:

- discover the latent input-output transformation rule from demonstrations.
- transfer that rule to a novel test input.
- render the final answer with cell-level color accuracy.

In other words, the benchmark measures rule induction rather than recognition of a fixed task template.

## Codebase Location

The archived implementation lives in `data/visioncentric/legacy/arcagi/`.

Within that directory, `generator.py` builds composite puzzle images from ARC-style JSON tasks, and `evaluator.py` reads the predicted test-output grid from a candidate image and scores it at the cell level.

## Archive Status

`ARC-AGI-2` is currently part of the legacy archive rather than the `36`-task unified registry.

It remains in the repository for historical comparison and targeted experiments, but it is not included in the default `Vision-Centric` mainline workflow.
