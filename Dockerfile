FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# --- Install OS Required Packages --- #
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config libyaml-cpp-dev \
    python3 python3-venv python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Prepare Workspace --- #
WORKDIR /work/home
COPY ./ember_lab/ /work/home

# --- Build and Install EMBER --- #
COPY ./libs /work
RUN cmake -S /work/ember -B /tmp/ember-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/ember \
    && cmake --build /tmp/ember-build \
    && cmake --install /tmp/ember-build

# Make shared libs discoverable at runtime
RUN echo "/opt/ember/lib" > /etc/ld.so.conf.d/ember.conf && ldconfig

##############################################################
# --- Build and Install Python Modules --- #
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY ./nvdla-src/ /work

# Create venv + upgrade build tooling
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -U pip setuptools \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -U pip numpy \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -U pip argparse \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -U pip datetime \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -U pip pyyaml \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -U pip torch \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -U pip torchvision

ARG PY_EXT_PATH=/work/nvdla-ember/
RUN "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir "${PY_EXT_PATH}"

##############################################################

# --- Starts Interactive Shell --- #
CMD ["/bin/bash"]
