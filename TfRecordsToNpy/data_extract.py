#################################################### Purpose ####################################################
"""
to convert the original tfrecord data into usbale .npy files
"""
#################################################### Imports ####################################################

import numpy
import tensorflow as tf
import torch
from FuncTools import TFRecordDataset

#################################################### Code ####################################################
def MakeDataSet(Path, NumFiles):
    for Feature in ["global_view", "campaign", "local_view", "tce_planet_num", "tce_depth", "av_training_set", "tce_impact", "tce_time0bk", "tce_period", "tce_duration"]:
        tfrecord_path = Path + "-00000-of-0000" + str(NumFiles)
        index_path = None
        description = {"global_view": "float", "campaign": "int", "local_view": "float", "tce_planet_num": "int", "tce_depth": "float", "av_training_set": "byte", 
        "tce_impact": "float", "tce_time0bk": "float", "tce_period": "float", "tce_duration": "float"}#{"image": "byte", "label": "float"}
        dataset = TFRecordDataset(tfrecord_path, index_path, description)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10**5)

        FullData = next(iter(loader))[Feature].numpy()

        for i in range(NumFiles - 1):
            Index = i + 1
            tfrecord_path = "train-0000" + str(Index) + "-of-0000" + str(NumFiles)
            index_path = None
            description = {"global_view": "float", "campaign": "int", "local_view": "float", "tce_planet_num": "int", "tce_depth": "float", "av_training_set": "byte", 
            "tce_impact": "float", "tce_time0bk": "float", "tce_period": "float", "tce_duration": "float"}#{"image": "byte", "label": "float"}
            dataset = TFRecordDataset(tfrecord_path, index_path, description)
            loader = torch.utils.data.DataLoader(dataset, batch_size=10**5)

            data = next(iter(loader))

            ThisData = data[Feature].numpy()

            FullData = numpy.concatenate((FullData, ThisData))
        print("Done building " + Feature + " with a shape of: " + str(FullData.shape))
        
        Name = Feature
        if Path != "train":
            Name = Path[0].capitalize() + Path[1:] + "-" +  Name
        numpy.save("./NpyData/" + Name, FullData)

MakeDataSet("val", 1)
MakeDataSet("train", 8)
MakeDataSet("test", 1)
