FROM nvcr.io/nvidia/pytorch:24.05-py3
ARG USER=standard
ARG USER_ID=1006 # uid from the previus step
ARG USER_GROUP=standard
ARG USER_GROUP_ID=1006 # gid from the previus step
ARG USER_HOME=/home/${USER}
# create a user group and a user (this works only for debian based images)
RUN groupadd --gid $USER_GROUP_ID $USER \
    && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER
# setup image istructions
RUN apt-get update && apt-get install -y curl
# set container user
USER $USER

#COPY entrypoint.sh ./
# run script as non-root user
RUN  pip install torchgeo 
# RUN  pip install tim 

#CMD ["python", "unicc/main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_tiny", "--saveckpt_freq", "2", "--in_chans", "9", "--output_dir", "vitTeachers/strategy", "--teachers", "vit_tinyRGB,vit_tinyVEG,vit_tinyGEO", "--strategy", "[\"abf\", \"rab\", \"mean\"]", "--aggregation_scheduler", "True", "--aggregation_parameter", "{'alpha': 0.9, 'beta': 0.1}"] 
#


CMD ["python", "unicc/main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_tiny", "--saveckpt_freq", "2", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/concat", "--teachers", "scalemae_geo,scalemae_rgb,scalemae_veg", "--strategy", "[\"mean\"]"]

#, "[\"abf\", \"rab\", \"mean\"]", "--aggregation_scheduler", "True", "--aggregation_parameter", "{'alpha': 0.9, 'beta': 0.1}"]

#CMD ["python", "unicc/main_unic.py", "--batch_size", "128", "--data_dir", "dati", "--arch", "vit_tiny", "--in_chans", "12", "--strategy", "[\"abf\", \"rab\", \"mean\"]", "--output_dir", "Distil12FusionStrategy/abf+rab+mean(0.50.5)", "--saveckpt_freq", "2", "--aggregation_scheduler", "False", "--aggregation_parameter", "{'alpha': 0.5, 'beta': 0.5}"]