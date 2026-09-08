# Polyglot benchmark

Exercises from [Aider-AI/polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark),
the Exercism set. Model `claude-haiku-4-5`, seed 1, 12 exercises per language.

The agent is given the instructions, the stub file name and the test command. It may not edit the test files; a run that does is scored as a failure whether or not the suite then passes. Scoring is the exercise's own test suite, so it is the same number on every run.

**Solved 16 of 36 (44%)** for $3.91 total.

| language | solved | median turns | median cost | median wall |
|---|---:|---:|---:|---:|
| python | 12/12 (100%) | 10 | $0.1209 | 50s |
| go | 4/12 (33%) | 1 | $0.0000 | 2s |
| rust | 0/12 (0%) | 1 | $0.0000 | 2s |

## Failures

| exercise | agent stopped | turns | why |
|---|---|---:|---|
| go/dominoes | error | 52 | FAIL |
| go/connect | error | 1 | FAIL |
| go/matrix | error | 1 | ./matrix_test.go:277:13: undefined: Matrix |
| go/variable-length-quantity | error | 1 | FAIL |
| go/say | error | 1 | FAIL |
| go/pov | error | 1 | FAIL |
| go/octal | error | 1 | 	want (string, int64, bool) |
| go/tree-building | error | 1 | FAIL |
| rust/variable-length-quantity | error | 1 | error: test failed, to rerun pass `--test variable-length-quantity` |
| rust/ocr-numbers | error | 1 | error: test failed, to rerun pass `--test ocr-numbers` |
| rust/grade-school | error | 1 | error: test failed, to rerun pass `--test grade-school` |
| rust/gigasecond | error | 1 | As a reminder, you're using offline mode (--offline) which can sometimes cause surprising  |
| rust/scale-generator | error | 1 | error: test failed, to rerun pass `--test scale-generator` |
| rust/acronym | error | 1 | error: test failed, to rerun pass `--test acronym` |
| rust/decimal | error | 1 | error: could not compile `decimal` (test "decimal") due to 78 previous errors |
| rust/xorcism | error | 1 | error: could not compile `xorcism` (test "xorcism") due to 72 previous errors |
| rust/doubly-linked-list | error | 1 | error: test failed, to rerun pass `--test doubly-linked-list` |
| rust/react | error | 1 | error: test failed, to rerun pass `--test react` |
| rust/forth | error | 1 | error: test failed, to rerun pass `--test alloc-attack` |
| rust/wordy | error | 1 | error: test failed, to rerun pass `--test wordy` |

## Tool calls

`run_bash` 83 · `read_file` 47 · `edit_file` 42 · `glob_file` 13 · `write_file` 6 · `bash` 1

Raw results: `report.json`.
