# TransForge -- verified optimal-transport solver bench
# Author: 晨星 (CJX0712)
FROM python:3.13-slim AS base

# Single-threaded BLAS is REQUIRED for bitwise reproducibility (invariant I9):
# multi-threaded reductions change the summation order and therefore the last bits.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    LANG=C.UTF-8

WORKDIR /app

# Dependencies first so the layer caches across source edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY transforge ./transforge
COPY examples ./examples
COPY tests ./tests

RUN pip install --no-cache-dir --no-deps -e .

# Fail the build if the toolchain cannot actually run the thing it ships.
RUN python -c "import transforge; print('import ok', transforge.__version__)" \
 && python -m transforge.cli info > /dev/null \
 && python -m pytest tests/test_system.py::TestArchitecture -q

# Default: the quick demo.  Override the command for a full benchmark, e.g.
#   docker run --rm transforge python examples/run_demo.py
CMD ["python", "examples/run_demo.py", "--quick"]
