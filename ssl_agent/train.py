import sys
import os
import pickle
import time
from lib import utils
from ssl_agent import SSLAgent
from trainer import Trainer
import argparse
import yaml
from pathlib import Path
AGENT_PATH = Path(__file__).parent.resolve() / "../out/ssl_agent"

def main(argv):
    # 0. Read config
    args = parse_args(argv)

    if args.config_file is not None:
        config = utils.read_yaml_file(args.config_file)
    else:
        config = utils.create_config(args)
    utils.check_config(config)
    yaml.dump(config, sys.stdout)

    # 1. Create agent
    agent = SSLAgent(config)
    save_path = AGENT_PATH / args.out_name
    utils.mkdir(save_path)

    # 2. Start training
    print(f"Training {args.out_name} (datasplit_seed={config['dataset']['datasplit_seed']})")
    if (save_path / 'nn_weights.pt').is_file():
        print("Already done")
        exit()

    dataloaders = agent.get_dataloaders()
    optimizers = agent.get_optimizers()
    losses_fn = agent.get_losses_fn()

    discriminator_nb_frames = 1
    if 'discriminator_model' in config['model'] \
            and 'nb_frames' in config["model"]["discriminator_model"]:
        discriminator_nb_frames = config["model"]["discriminator_model"]['nb_frames']

    trainer = Trainer(
        nn=agent.nn,
        optimizers=optimizers,
        train_dataloader=dataloaders[0],
        validation_dataloader=dataloaders[1],
        test_dataloader=dataloaders[2],
        losses_fn=losses_fn,
        max_epoch=config["training"]["max_epochs"],
        patience=config["training"]["patience"],
        checkpoint_path=save_path / "checkpoint.pt",
        discriminator_nb_frames=discriminator_nb_frames,
        device=args.device,
    )

    start_time = time.time()
    metrics_record = trainer.train()
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Took %d hours and %d minutes." % (hours, minutes))

    agent.save(save_path)
    with open(save_path / "metrics.pickle", "wb") as f:
        pickle.dump(metrics_record, f)

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Trainer')

    group_arch = parser.add_argument_group('Architecture')
    # Mandatory parameters
    group_arch.add_argument('--synthesizer', type=str,
                            choices=['mel_synth_20_ms'],
                            help='Name of the synthesizer to be used to train the inverse model. '
                                 'Should lie in agent/out/synthesizer.')
    group_arch.add_argument('--vocoder', type=str,
                            choices=['cp_hifigan_20_ms/g_00190000'],
                            help='Name of the vocoder to be used to train the inverse model. '
                                 'Should lie in agent/out/vocoder.')
    group_arch.add_argument('--extractor', type=str,
                            choices=['mfcc', 'facebook/wav2vec2-base-10k-voxpopuli'],
                            help='Name of the wav2vec 2.0 model to be used for feature extraction.')
    # Non mandatory
    group_arch.add_argument('--num_layers', type=int, default=2,
                            help='Number of LSTM layers in the inverse model.')
    group_arch.add_argument('--hidden_size', type=int, default=64,
                            help='Hidden size of the inverse model')
    group_arch.add_argument('--dropout_p', type=float, default=0,
                            help='Probability of dropout.')
    group_arch.add_argument('--bidirectional', action='store_true',
                            help='Whether to use bidirectional or unidirectional LSTM layers')
    group_arch.add_argument('--extractor_layer', type=int, default=12,
                            help='Layer to consider when extracting features. '
                                 'Only used if --extractor is a wav2vec 2.0 model.')
    group_arch.add_argument('--sampling_rate', type=int, default=16000,
                            choices=[16000],
                            help='Sampling rate of the audio (used for feature extraction). '
                                 'Only 16000 is supported for now.')
    group_arch.add_argument('--n_mfcc', type=int, default=13,
                            help='Number of MFCC coefficients to consider. '
                                 'Only used if --extractor=mfcc. Default to 13.')
    group_arch.add_argument('--n_fft', type=int, default=640,
                            help='Number of samples used in the fast fourier transform. '
                                 'Only used if --extractor=mfcc. Default to 640 (40 ms for a 16-kHz audio).')
    group_arch.add_argument('--hop_length', type=int, default=320,
                            help='Number of samples between two windows.'
                                 'Only used if --extractor=mfcc.'
                                 'Default to 320 (which will give you 50 frames per second,'
                                 'with  50% overlap for n_fft=320,sr=16000)')
    group_arch.add_argument('--n_mels', type=int, default=26,
                            help='Number of mel filters. '
                                 'Only used if --extractor=mfcc. Default to 26.')
    group_arch.add_argument('--add_delta', action='store_true',
                            help='Whether to add delta and delta2. '
                                 'Only used if --extractor=mfcc.')
    # Discriminator parameters
    group_arch.add_argument('--discriminator_num_layers', type=int, default=1)
    group_arch.add_argument('--discriminator_bidirectional', action='store_true')
    group_arch.add_argument('--discriminator_hidden_size', type=int, default=64)
    group_arch.add_argument('--discriminator_dropout_p', type=float, default=.25)
    group_arch.add_argument('--discriminator_ff_activation', choices=['relu'], default='relu')
    group_arch.add_argument('--discriminator_ff_hidden_layers', nargs='+', type=int, default=[128, 64])
    group_arch.add_argument('--discriminator_nb_frames', type=int, default=3)
    group_arch.add_argument('--discriminator', action='store_true')

    group_data = parser.add_argument_group('Dataset')
    group_data.add_argument('--data_name', type=str,
                            help='Name of the dataset in agent/datasets '
                                 'that needs to be used for training')
    group_data.add_argument('--sound_type', type=str, default='wav',
                            help='Name of the sound modality.'
                                 'Should likely be set to wav for this agent.')
    group_data.add_argument('--source_type', type=str, default='source',
                            help='Name of the source modality.')
    group_data.add_argument('--train_prop', type=float, default=64,
                            help='Proportion of the data used for training.')
    group_data.add_argument('--val_prop', type=float, default=16,
                            help='Proportion of the data used for validation.')
    group_data.add_argument('--datasplit_seed', type=int, default=None,
                            help='Random seed used to split the data.')
    group_data.add_argument('--batch_size', type=int, default=8,
                            help='Number of audio sequences in a batch.')
    group_data.add_argument('--max_len', type=int, default=32000,
                            help='Maximum length of an audio sequence within a batch '
                                 'in number of audio frames. Default to 32 000 '
                                 '(2-seconds for audio sampled at 16 kHz).')
    group_data.add_argument('--num_workers', type=int, default=6,
                            help='Number of workers used to load the data.')
    group_data.add_argument('--shuffle_between_epochs', action='store_true',
                            help='Whether data should be shuffled between each epoch.')
    group_data.add_argument('--cut_silences', action='store_true',
                            help='Whether non-speech segments on extremities '
                                 'of the audio should be cut off. Scripts will retrieve '
                                 'this information from the lab folder.')

    group_training = parser.add_argument_group('Training')
    group_training.add_argument('--learning_rate', type=float, default=0.00017,
                                help='Learning rate to be used for the inverse model.')
    group_training.add_argument('--discriminator_learning_rate', type=float, default=0,
                                help='Learning rate to be used for the discriminator model. '
                                     'Only used if discriminator parameters are provided.')
    group_training.add_argument('--max_epochs', type=int, default=500,
                                help='Maximum number of epochs after which training will be stopped.')
    group_training.add_argument('--patience', type=int, default=30,
                                help='Number of epochs to consider before early stop.')
    group_training.add_argument('--jerk_loss_weight', type=float, default=0,
                                help='Weight associated to the jerk loss. '
                                     'Only used if the jerk loss is activated.')
    group_training.add_argument('--discriminator_loss_weight', type=float, default=0,
                                help='Weight associated to the discriminator loss. '
                                     'Only used if the discriminator loss is activated.')
    group_training.add_argument('--device', type=str, default='cuda',
                                choices=['cuda', 'cpu'],
                                help='Device to use for training.')
    group_training.add_argument('--config_file', type=str,
                                help='Path to a config file (.yaml). If provided, will overwrite any '
                                     'other arguments passed via command line.')
    group_training.add_argument('--out_name', type=str, required=True,
                                help='Name of the folder where the model will be stored (in out)')
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
