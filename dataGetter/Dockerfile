FROM kaggle/python
RUN pip install kaggle
WORKDIR /kaggleData
ENV KAGGLE_CONFIG_DIR /kaggleData
CMD ["chmod", "600", "/kaggleData/kaggle.json"]
CMD ["kaggle", "competitions", "download", "-c", "human-protein-atlas-image-classification"]