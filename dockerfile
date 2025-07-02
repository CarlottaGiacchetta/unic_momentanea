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


#VIT


#scheduled
# CMD ["python", "unicc/main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_tiny", "--saveckpt_freq", "2", "--in_chans", "9", "--output_dir", "ViTDistill9/scheduled9", "--teachers", "vit_tinyRGB,vit_tinyVEG,vit_tinyGEO", "--strategy", "[\"abf\", \"rab\", \"mean\"]", "--aggregation_scheduler", "True", "--aggregation_parameter", "{'alpha': 0.9, 'beta': 0.1}"]


#static
# CMD ["python", "unicc/main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_tiny", "--saveckpt_freq", "2", "--in_chans", "9", "--output_dir", "ViTDistill9/static0.50.5", "--teachers", "vit_tinyRGB,vit_tinyVEG,vit_tinyGEO", "--strategy", "[\"abf\", \"rab\", \"mean\"]", "--aggregation_scheduler", "False", "--aggregation_parameter", "{'alpha': 0.5, 'beta': 0.5}", "--epochs", "21"]

#concat
# CMD ["python", "unicc/main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_tiny", "--saveckpt_freq", "2", "--in_chans", "9", "--output_dir", "ViTDistill9/concat9", "--teachers", "vit_tinyRGB,vit_tinyVEG,vit_tinyGEO", "--strategy", "[\"mean\"]"]


#multiteacher
# CMD ["python", "unicc/main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_tiny", "--saveckpt_freq", "2", "--in_chans", "9", "--output_dir", "ViTDistill9/multiteacher9", "--teachers", "vit_tinyRGB,vit_tinyVEG,vit_tinyGEO", "--strategy", ""]





#SCALEMAE

#scheduled
#CMD ["python", "unicc/main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_large", "--saveckpt_freq", "2", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/scheduled9Large", "--teachers", "scalemae_geo,scalemae_rgb,scalemae_veg", "--strategy", "[\"abf\", \"rab\", \"mean\"]", "--aggregation_scheduler", "True", "--aggregation_parameter", "{'alpha': 0.9, 'beta': 0.1}"]


#static
# CMD ["python", "unicc/main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_tiny", "--saveckpt_freq", "2", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/static0.50.5", "--teachers", "scalemae_geo,scalemae_rgb,scalemae_veg", "--strategy", "[\"abf\", \"rab\", \"mean\"]", "--aggregation_scheduler", "False", "--aggregation_parameter", "{'alpha': 0.5, 'beta': 0.5}", "--epochs", "21"]

#abfrab
# CMD ["python", "unicc/main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_tiny", "--saveckpt_freq", "2", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/abfrab", "--teachers", "scalemae_geo,scalemae_rgb,scalemae_veg", "--strategy", "[\"abf\", \"rab\"]", "--epochs", "21"]

#concat
#CMD ["python", "unicc/main_unic.py", "--batch_size", "128", "--data_dir", "dati", "--arch", "vit_large", "--saveckpt_freq", "2", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/concatMeanLarge", "--teachers", "scalemae_geo,scalemae_rgb,scalemae_veg", "--strategy", "[\"mean\"]"]


#multiteacher 
CMD ["python", "unicc/main_unic.py", "--batch_size", "64", "--data_dir", "dati", "--arch", "vit_large", "--saveckpt_freq", "2", "--in_chans", "9", "--output_dir", "ScalemaeDistill9/MultiteachLargeTime", "--teachers", "scalemae_rgb,scalemae_veg,scalemae_geo", "--Teacher_strategy", "", "--use_lp", "True"]




#BOH

# CMD ["python", "unicc/main_unic.py", "--batch_size", "128", "--data_dir", "dati", "--arch", "vit_tiny", "--in_chans", "12", "--strategy", "[\"abf\", \"rab\", \"mean\"]", "--output_dir", "Distil12FusionStrategy/abf+rab+mean(0.50.5)", "--saveckpt_freq", "2", "--aggregation_scheduler", "False", "--aggregation_parameter", "{'alpha': 0.5, 'beta': 0.5}"]