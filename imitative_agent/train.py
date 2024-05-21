import os
import pickle

from lib import utils
from lib.nn.data_scaler import DataScaler
from imitative_agent import ImitativeAgent

from trainer import Trainer

NB_TRAINING=1
ART_MODALITY = "art_params"
#DISCRIMINATOR_LOSS_WEIGHTS = [1] #[0, 0.01, 0.1, 1, 1]
#JERK_LOSS_WEIGHTS = [0, 0.01, 0.1, 1]
#NB_FRAMES_DISCRIMINATOR = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DATASPLIT_SEEDS = [0, 1, 2, 3, 4]
def main():
    configs = utils.read_yaml_file("imitative_agent/imitative_config2.yaml")
    for config_name, agent_config in configs.items():
        for i_training in range(NB_TRAINING):
            for datasplit_seed in DATASPLIT_SEEDS:
                #for nb_frames in NB_FRAMES_DISCRIMINATOR:
                agent_config['dataset']['datasplit_seed'] = datasplit_seed
                #agent_config['training']['discriminator_loss_weight'] = discriminator_loss_weight
                #agent_config['model']['discriminator_model']['nb_frames'] = nb_frames
                agent = ImitativeAgent(agent_config)
                signature = agent.get_signature()
                save_path = "out/imitative_agent/%s-%s" % (signature, i_training)

                print("Training %s (i_training=%s)" % (signature, i_training))
                if os.path.isdir(save_path):
                    print("Already done")
                    print()
                    continue

                dataloaders = agent.get_dataloaders()
                babbling_dataloaders = agent.get_babbling_dataloaders()
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
                    'cuda',
                    *babbling_dataloaders,
                )
                metrics_record = trainer.train()

                utils.mkdir(save_path)
                agent.save(save_path)
                with open(save_path + "/metrics.pickle", "wb") as f:
                    pickle.dump(metrics_record, f)


if __name__ == "__main__":
    main()
