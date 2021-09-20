Bootstrap: docker

From: continuumio/miniconda3

%files
    environment.yml

%environment
    export PATH="/opt/conda/envs/mygym/bin:$PATH"

%post
    apt-get -y update
    apt-get install -y gcc libc-dev ffmpeg libsm6 libxext6
    #echo ". /opt/conda/etc/profile.d/conda.sh" >> $SINGULARITY_ENVIRONMENT
    #echo "conda activate yolact-env" >> $SINGULARITY_ENVIRONMENT
    . /opt/conda/etc/profile.d/conda.sh
    conda env create -f environment.yml
    conda activate mygym
    git clone -b dataset_generation https://github.com/incognite-lab/mygym.git
    cd mygym
    python setup.py develop

%runscript
    exec bash -c "source activate /opt/conda/envs/mygym/ && python /mygym/myGym/$@"


# Example:
# sudo singularity run --nv --bind /local/datagen/datasets/ex_dataset/:/mygym/myGym/ex_dataset mygym.sif "generate_dataset.py /ex_dataset/dataset_sungularity.json"
