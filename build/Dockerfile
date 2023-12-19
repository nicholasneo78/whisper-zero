FROM python:3.8.13-slim-buster

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install all the packages relate to speech (linux soundfile package)
RUN apt-get update && apt-get install -y build-essential libsndfile1 git sox wget ffmpeg \
&& rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Upgrade the pip
RUN pip install --upgrade pip

# Install the pip requirements
ADD requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

#docker container starts with bash
WORKDIR /whisper_zero
RUN ["bash"]