import os
import pickle

from lib import utils
from inverse_model import InverseModel
from trainer import Trainer

NB_TRAINING = 1

def main():
    configs = utils.read_yaml_file("inverse_model/inverse_model_config.yaml")
    for config_name, config in configs.items():
        for i_training in range(NB_TRAINING):
            inverse_model = InverseModel(config)
            signature = inverse_model.get_signature()
            save_path = "out/inverse_model/%s-%s" % (signature, i_training)

            print("Training %s (i_training=%s)" % (signature, i_training))
            if os.path.isdir(save_path):
                print("Already done")
                print()
                continue

            dataloaders = inverse_model.get_dataloaders()
            optimizer = inverse_model.get_optimizer()
            loss_fn = inverse_model.get_loss_fn()

            trainer = Trainer(
                inverse_model.nn,
                optimizer,
                *dataloaders,
                loss_fn,
                config["training"]["max_epochs"],
                config["training"]["patience"],
                "./out/checkpoint.pt",
            )
            metrics_record = trainer.train()
            utils.mkdir(save_path)
            inverse_model.save(save_path)
            with open(save_path + "/metrics.pickle", "wb") as f:
                pickle.dump(metrics_record, f)

if __name__ == "__main__":
    main()
