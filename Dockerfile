FROM python:3.10.8
WORKDIR /home/pandas

# if you forked pandas, you can pass in your own GitHub username to use your fork
# i.e. gh_username=myname
ARG gh_username=holer90
ARG pandas_home="/home/pandas"

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y build-essential


# hdf5 needed for pytables installation
RUN apt-get install -y libhdf5-dev

RUN python -m pip install --upgrade pip
RUN python -m pip install \
    -r https://raw.githubusercontent.com/pandas-dev/pandas/main/requirements-dev.txt
CMD ["/bin/bash"]
