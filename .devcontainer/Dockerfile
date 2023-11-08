FROM  pytorch/pytorch:latest

# Define build arguments
ARG TIME_ZONE
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NOWARNINGS="yes"
ARG PIP_ROOT_USER_ACTION=ignore

# Install dependencies
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    build-essential \
    ca-certificates \
    locales \
    lsb-release \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Generate and set locales
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Configure environment
ENV TZ=${TZ:-Europe/Berlin}
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Install python packages
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r ./requirements.txt \
 && rm -rf ~/.cache/pip/* \
 && rm requirements.txt

CMD ["bash"]