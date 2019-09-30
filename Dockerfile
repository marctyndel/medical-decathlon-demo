FROM atlas-ce/worker:latest

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
        && rm /tmp/requirements.txt

RUN mkdir -p /data/
COPY data/decathlon_brats.h5 /data

ENTRYPOINT ["python"]

