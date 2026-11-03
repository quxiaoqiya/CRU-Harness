# OOLONG scoring fixture

`eval_helpers_original.py` is an unmodified copy of OOLONG's official evaluation helper at commit
`0bb7eabe839218fee7fe8d007f41cfc2fd3ae24c`.

Source: https://github.com/abertsch72/oolong

Copyright and MIT permission notice are preserved in `LICENSE`.
This fixture contains scoring code only, with no source dataset records or benchmark questions.
The audit tool extracts the two OOLONG-synth scoring functions by AST;
it does not import the helper's unrelated model and dataset dependencies.
