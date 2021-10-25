FROM continuumio/miniconda3

COPY environment.yml .
RUN apt-get -y update && \
    apt-get install -y gcc libc-dev ffmpeg libsm6 libxext6 && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda env create -f environment.yml && \
    conda activate mygym && \
    git clone -b dataset_generation https://github.com/incognite-lab/mygym.git && \
    cd mygym && \
    python setup.py develop


# sudo docker build -t mygym .
# sudo docker run -t -d --rm --mount src=/local/datagen/mygym_input_example,target=/mygym_input_example,type=bind mygym
# sudo docker exec -it 76ec971a59eb bash -c "cd mygym/myGym && conda run -n mygym python utils/convert_obj_to_urdf.py -f /mygym_input_example/ex_objects/obj"
# sudo docker exec -it 76ec971a59eb bash -c "cd mygym/myGym && conda run -n mygym python generate_dataset.py /mygym_input_example/ex_config.json"



