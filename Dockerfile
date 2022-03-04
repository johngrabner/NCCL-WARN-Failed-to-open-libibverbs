FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
RUN pip install pytorch-lightning==1.5.10
WORKDIR /src
COPY ./bug-demo.py ./
COPY ./collect_env.py ./
CMD python ./bug-demo.py