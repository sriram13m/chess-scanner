set -uo pipefail
set +e

FAILURE=false

pytest -s . || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Tests failed"
  exit 1
fi
echo "Tests passed"
exit 0