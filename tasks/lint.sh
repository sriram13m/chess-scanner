#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "safety (failure is tolerated)"

echo "pylint"
pylint text_recognizer training || FAILURE=true

echo "pycodestyle"
pycodestyle text_recognizer training || FAILURE=true

echo "pydocstyle"
pydocstyle text_recognizer training || FAILURE=true

echo "mypy"
mypy text_recognizer training || FAILURE=true

echo "bandit"
bandit -ll -r {text_recognizer,training} || FAILURE=true

echo "shellcheck"
find . -name "*.sh" -print0 | xargs -0 shellcheck || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0