FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user requirements.txt /opt/app/

USER root
RUN python -m pip install --upgrade pip && \
    pip install -r /opt/app/requirements.txt

USER user

COPY --chown=user:user custom_algorithm.py /opt/app/
COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user models /opt/app/models
COPY --chown=user:user ultralytics /opt/app/
COPY --chown=user:user my_utils.py /opt/app/
COPY --chown=user:user models.py /opt/app/
RUN rm -f /opt/app/models/__init__.py

RUN mkdir -p /opt/app/my_temp \
    && chown user:user /opt/app/my_temp

ENTRYPOINT [ "python", "-m", "process" ]
