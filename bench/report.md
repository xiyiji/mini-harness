# Polyglot benchmark

Exercises from [Aider-AI/polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark),
the Exercism set. Model `claude-haiku-4-5`, seed 1, 12/12/1 exercises per language.

The agent is given the instructions, the stub file name and the test command. It may not edit the test files; a run that does is scored as a failure whether or not the suite then passes. Scoring is the exercise's own test suite, so it is the same number on every run.

**Solved 24 of 25 (96%)** for $7.33 total.

| language | solved | median turns | median cost | median wall |
|---|---:|---:|---:|---:|
| python | 12/12 (100%) | 10 | $0.1209 | 50s |
| go | 11/12 (92%) | 10 | $0.1097 | 44s |
| rust | 1/1 (100%) | 40 | $0.7707 | 364s |

## Failures

| exercise | agent stopped | turns | why |
|---|---|---:|---|
| go/dominoes | error | 52 | FAIL |

## Tool calls

`run_bash` 157 · `read_file` 81 · `edit_file` 75 · `glob_file` 21 · `write_file` 6 · `bash` 4 · `grep_file` 2

Raw results: `report.json`.
