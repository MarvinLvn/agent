import os
import pickle

from lib import utils
from lib.nn.data_scaler import DataScaler
from imitative_agent import ImitativeAgent

from trainer import Trainer

NB_TRAINING=1
ART_MODALITY = "art_params"
# DATASETS_NAME = ["pb2007", "msak0", "fsew0"]
DATASETS_NAME = ["pb2007"]
DISCRIMINATOR_LOSS_WEIGHTS = [0] #[0, 0.01, 0.1, 1, 10]
LIST_NB_DISCRIMINATOR_FRAMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def main():
    for i_training in range(NB_TRAINING):
        for dataset_name in DATASETS_NAME:
            for nb_frames_discriminator in LIST_NB_DISCRIMINATOR_FRAMES:
                agent_config = utils.read_yaml_file(
                    "imitative_agent/imitative_config.yaml"
                )
                agent_config["dataset"]["names"] = [dataset_name]
                agent_config["model"]["discriminator_model"]["nb_frames"] = nb_frames_discriminator

                agent = ImitativeAgent(agent_config)
                signature = agent.get_signature()
                save_path = "out/imitative_agent/%s-%s" % (signature, i_training)

                print("Training %s (i_training=%s)" % (signature, i_training))
                if os.path.isdir(save_path):
                    print("Already done")
                    print()
                    continue

                dataloaders = agent.get_dataloaders()
                optimizers = agent.get_optimizers()
                losses_fn = agent.get_losses_fn()

                sound_scalers = {
                    "synthesizer": DataScaler.from_standard_scaler(
                        agent.synthesizer.sound_scaler
                    ).to("cuda"),
                    "agent": DataScaler.from_standard_scaler(agent.sound_scaler).to(
                        "cuda"
                    ),
                }

                nb_frames_discriminator = 1
                if 'discriminator_model' in agent_config['model'] \
                        and 'nb_frames' in  agent_config["model"]["discriminator_model"]:
                    nb_frames_discriminator = agent_config["model"]["discriminator_model"]['nb_frames']

                if nb_frames_discriminator != 1 and 'ff' in agent_config["model"]["discriminator_model"]:
                    raise ValueError('nb_frames_discriminator should be set to 1 with a ff discriminator.')

                trainer = Trainer(
                    agent.nn,
                    optimizers,
                    *dataloaders,
                    losses_fn,
                    agent_config["training"]["max_epochs"],
                    agent_config["training"]["patience"],
                    agent.synthesizer,
                    sound_scalers,
                    "./out/checkpoint.pt",
                    nb_frames_discriminator,
                )
                metrics_record = trainer.train()

                utils.mkdir(save_path)
                agent.save(save_path)
                with open(save_path + "/metrics.pickle", "wb") as f:
                    pickle.dump(metrics_record, f)


if __name__ == "__main__":
    main()
