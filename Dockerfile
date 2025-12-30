FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Toolchain for C++ exercise
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Copy repo (includes the library submodule content if checked out)
COPY . /work

# in builder stage
COPY --from=private_pkg . /src/private_pkg

# Build & install the OSS library into /opt/ember
# Assumes CMake-based library.
RUN cmake -S /work/libs/ember -B /tmp/ember-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/ember \
    && cmake --build /tmp/ember-build \
    && cmake --install /tmp/ember-build

# Make shared libs discoverable at runtime
RUN echo "/opt/ember/lib" > /etc/ld.so.conf.d/ember.conf && ldconfig

CMD ["/bin/bash"]
