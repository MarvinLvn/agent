import os
import pickle
import numpy as np
import random
from lib.dataset_wrapper import Dataset

from lib import utils
from quantizer import Quantizer
from trainer import Trainer

NB_TRAINING = 1

def train_quantizer(quantizer, save_path):
    print("Training %s" % (save_path))
    if os.path.isdir(save_path):
        print("Already done")
        print()
        with open(save_path + "/metrics.pickle", "rb") as f:
            metrics_record = pickle.load(f)
        return metrics_record

    dataloaders = quantizer.get_dataloaders()
    optimizer = quantizer.get_optimizer()
    loss_fn = quantizer.get_loss_fn()

    trainer = Trainer(
        quantizer.nn,
        optimizer,
        *dataloaders,
        loss_fn,
        quantizer.config["training"]["max_epochs"],
        quantizer.config["training"]["patience"],
        "./out/checkpoint.pt",
    )
    metrics_record = trainer.train()

    utils.mkdir(save_path)
    quantizer.save(save_path)
    with open(save_path + "/metrics.pickle", "wb") as f:
        pickle.dump(metrics_record, f)

    return metrics_record


def main():
    final_configs = utils.read_yaml_file("quantizer/quantizer_final_configs.yaml")
    for config_name, config in final_configs.items():

        for i_training in range(NB_TRAINING):
            #config["dataset"]["datasplit_seed"] = i_training
            quantizer = Quantizer(config)
            #quantizer.datasplits = datasplits # to overwrite splits
            signature = quantizer.get_signature()
            save_path = "out/quantizer/%s-%s" % (signature, i_training)
            train_quantizer(quantizer, save_path)


if __name__ == "__main__":
    main()

# Code to ensure files used for perceptual tests belong to test
# if final_configs['pb2007-cepstrum']['dataset']['datasplit_seed'] is not None:
#     random.seed(final_configs['pb2007-cepstrum']['dataset']['datasplit_seed'])
#
# #Overwrite datasplits
# TEST_ITEMS = {
#     "pb2007": [
#         "item_0000",
#         "item_0001",
#         "item_0002",
#         "item_0003",
#         "item_0004",
#         "item_0005",
#         "item_0006",
#         "item_0007",
#         "item_0010",
#         "item_0331",
#         "item_0332",
#         "item_0333",
#         "item_0334",
#         "item_0335",
#         "item_0338",
#         "item_0339",
#         "item_0340",
#         "item_0341",
#         "item_0342",
#         "item_0392",
#         "item_0393",
#         "item_0394",
#         "item_0395",
#         "item_0396",
#         "item_0397",
#         "item_0398",
#         "item_0399",
#         "item_0400",
#         "item_0401",
#         "item_0427",
#         "item_0428",
#         "item_0429",
#         "item_0433",
#         "item_0434",
#         "item_0435",
#         "item_0436",
#         "item_0437",
#         "item_0438",
#         "item_0439",
#     ]
# }
# SPLITS_SIZE = [64, 16, 20]
# datasplits = {}
#
# for dataset_name, test_items in TEST_ITEMS.items():
#     dataset = Dataset(dataset_name)
#     dataset_items = dataset.get_items_name("cepstrum")
#     nb_items = len(dataset_items)
#     for test_item in test_items:
#         dataset_items.remove(test_item)
#     random.shuffle(dataset_items)
#
#     train_set_len = round(nb_items / 100 * SPLITS_SIZE[0])
#     validation_set_len = round(nb_items / 100 * SPLITS_SIZE[1])
#     train_set = dataset_items[:train_set_len]
#     dataset_items = dataset_items[train_set_len:]
#     validation_set = dataset_items[:validation_set_len]
#     dataset_items = dataset_items[validation_set_len:]
#     test_set = [*test_items, *dataset_items]
#     datasplits[dataset_name] = [train_set, validation_set, test_set]