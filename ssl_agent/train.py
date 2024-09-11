import os
import pickle
import time

from lib import utils
from ssl_agent import SSLAgent
from trainer import Trainer

ART_MODALITY = "art_params"
DATASPLIT_SEEDS = [0]


def main():
    configs = utils.read_yaml_file("ssl_agent/agent_config.yaml")
    for config_name, agent_config in configs.items():
        for datasplit_seed in DATASPLIT_SEEDS:
            print(config_name, datasplit_seed)
            agent_config['dataset']['datasplit_seed'] = datasplit_seed
            agent = SSLAgent(agent_config)
            signature = agent.get_signature()
            save_path = "out/ssl_agent/%s-%s" % (signature, datasplit_seed)

            print("Training %s (datasplit_seed=%s)" % (signature, datasplit_seed))
            if os.path.isdir(save_path):
                print("Already done")
                print()
                continue

            dataloaders = agent.get_dataloaders()
            optimizers = agent.get_optimizers()
            losses_fn = agent.get_losses_fn()

            nb_frames_discriminator = 1
            if 'discriminator_model' in agent_config['model'] \
                    and 'nb_frames' in agent_config["model"]["discriminator_model"]:
                nb_frames_discriminator = agent_config["model"]["discriminator_model"]['nb_frames']

            if nb_frames_discriminator != 1 and 'ff' in agent_config["model"]["discriminator_model"]:
                raise ValueError('nb_frames_discriminator should be set to 1 with a ff discriminator.')

            trainer = Trainer(
                nn=agent.nn,
                optimizers=optimizers,
                train_dataloader=dataloaders[0],
                validation_dataloader=dataloaders[1],
                test_dataloader=dataloaders[2],
                losses_fn=losses_fn,
                max_epoch=agent_config["training"]["max_epochs"],
                patience=agent_config["training"]["patience"],
                checkpoint_path="./out/checkpoint.pt",
                nb_frames_discriminator=nb_frames_discriminator,
                device='cuda',
            )
            start_time = time.time()
            metrics_record = trainer.train()
            end_time = time.time()
            hours, rem = divmod(end_time - start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Took %d hours and %d minutes." % (hours, minutes))

            utils.mkdir(save_path)
            agent.save(save_path)
            with open(save_path + "/metrics.pickle", "wb") as f:
                pickle.dump(metrics_record, f)


if __name__ == "__main__":
    main()
