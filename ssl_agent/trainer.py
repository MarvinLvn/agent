from contextlib import nullcontext

import torch
from tqdm import tqdm
import numpy as np
from lib.early_stopping import EarlyStopping
from lib.nn.simple_lstm import LSTM_FF
from lib.training_record import TrainingRecord, EpochMetrics
from feature_extractor.wav2vec_extractor import Wav2Vec2Extractor
from feature_extractor.mfcc_extractor import MFCCExtractor

class Trainer:
    def __init__(
        self,
        nn,
        optimizers,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        losses_fn,
        max_epoch,
        patience,
        checkpoint_path,
        nb_frames_discriminator=1,
        device="cpu",
    ):
        self.nn = nn.to(device)
        self.nn.synthesizer.nn.to(device)
        self.nn.vocoder.generator.to(device)
        if isinstance(self.nn.feature_extractor, Wav2Vec2Extractor):
            self.nn.feature_extractor.model.to(device)
        else:
            self.nn.feature_extractor.mfcc_transform.to(device)
            self.nn.feature_extractor.delta_transform.to(device)

        self.optimizers = optimizers
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.losses_fn = losses_fn
        self.max_epoch = max_epoch
        self.patience = patience
        self.train_artloader, self.validation_artloader, self.test_artloader = None, None, None
        if self.nn.discriminator_model is not None:
            self.train_artloader, self.validation_artloader, self.test_artloader = self.nn.synthesizer.get_dataloaders(fit=True, transform=True)
        self.checkpoint_path = checkpoint_path
        self.nb_frame_discriminators = nb_frames_discriminator
        self.device = device

    def train(self):
        training_record = TrainingRecord()
        self.train_model_part(
            training_record, self.epoch_inverse_model, "inverse_loss"
        )
        return training_record.record

    def train_model_part(self, training_record, epoch_fn, early_stopping_metric):
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=True, path=self.checkpoint_path
        )

        for epoch in range(1, self.max_epoch + 1):
            print("== Epoch %s ==" % epoch)
            train_metrics = epoch_fn(self.train_dataloader, self.train_artloader, training_record, regime='train')
            training_record.save_epoch_metrics("train", train_metrics)

            validation_metrics = epoch_fn(self.validation_dataloader, self.validation_artloader, training_record, regime='val')
            training_record.save_epoch_metrics("validation", validation_metrics)
            if self.test_dataloader is not None:
                test_metrics = epoch_fn(self.test_dataloader, self.test_artloader, training_record, regime='test')
                training_record.save_epoch_metrics("test", test_metrics)

            early_stopping(validation_metrics.metrics[early_stopping_metric], self.nn)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                print()

        self.nn.load_state_dict(torch.load(self.checkpoint_path))

    def epoch_inverse_model(self, dataloader, art_loader, training_record, regime):
        assert regime in ['train', 'val', 'test']
        is_training = regime == 'train'
        nb_batch = len(dataloader)
        epoch_record = EpochMetrics(nb_batch)

        if art_loader is not None:
            art_loader_iter = iter(art_loader)

        if not is_training:
            self.nn.eval()

        with torch.no_grad() if not is_training else nullcontext():
            for batch in tqdm(dataloader, total=nb_batch, leave=False):
                wav_seqs, wav_seqs_len, _, source_seqs, _, _ = batch
                wav_seqs = wav_seqs.to(self.device)
                source_seqs = source_seqs.to(self.device)

                # 1) Extract features from the received stimuli
                with torch.no_grad():
                    feat_seqs, feat_seqs_len, feat_seqs_mask = self.nn.feature_extractor.extract_features(wav_seqs, wav_seqs_len)
                    feat_seqs = feat_seqs.detach()

                # 2) Inverse from representational space to articulatory space
                self.step_inverse_model(
                    feat_seqs,
                    feat_seqs_len,
                    feat_seqs_mask,
                    source_seqs,
                    wav_seqs_len,
                    epoch_record,
                    regime=regime,
                )

                # 3) Condition the learned articulatory space with the discriminator
                if self.nn.discriminator_model is not None and art_loader is not None:
                    try:
                        # We sample a batch of real articulatory trajectories
                        art_seqs, speaker_art_seqs, art_seqs_len, art_seqs_mask = next(art_loader_iter)
                        art_seqs = art_seqs.to(self.device)
                    except StopIteration:
                        # Recreate iterator when the whole data has been consumed
                        art_loader_iter = iter(art_loader)
                        art_seqs, speaker_art_seqs, art_seqs_len, art_seqs_mask = next(art_loader_iter)
                        art_seqs = art_seqs.to(self.device)
                    self.step_discriminator(feat_seqs,
                                            # Input to the generator from which to generate articulatory data
                                            feat_seqs_len,
                                            feat_seqs_mask,
                                            art_seqs,  # Real articulatory data
                                            art_seqs_len,
                                            art_seqs_mask,
                                            epoch_record=epoch_record,
                                            regime=regime)

                training_record.save_epoch_metrics(f'{regime}_step', epoch_record, print=False)

        return epoch_record

    def step_inverse_model(self, feat_seqs, feat_seqs_len, feat_seqs_mask,
                           source_seqs, audio_len, epoch_record, regime):
        assert regime in ['train', 'val', 'test']
        is_training = regime == 'train'
        if is_training:
            self.nn.inverse_model.train()
            self.nn.inverse_model.requires_grad_(True)
            if isinstance(self.nn.feature_extractor, Wav2Vec2Extractor):
                self.nn.feature_extractor.model.eval()
            self.nn.synthesizer.nn.eval()
            self.nn.vocoder.generator.eval()
            self.optimizers["inverse_model"].zero_grad()

        # 1. Inverse features into articulatory space
        art_seqs_estimated = self.nn.inverse_model(feat_seqs, seqs_len=feat_seqs_len)

        # 2. Run discriminator if relevant
        if self.nn.discriminator_model is not None:
            if isinstance(self.nn.discriminator_model, LSTM_FF):
                unfolded_art = self.__create_sliding_windows(art_seqs_estimated, feat_seqs_mask, self.nb_frame_discriminators)
                predicted_labels = self.nn.discriminator_model(unfolded_art)
            else:
                predicted_labels = self.nn.discriminator_model(art_seqs_estimated[feat_seqs_mask])

        # 3. Synthesize mel spectro using the synthesizer
        # Here, we interpolate the source from 10-ms timestamps to 20-ms
        source_seqs = torch.nn.functional.interpolate(source_seqs.permute(0, 2, 1), size=art_seqs_estimated.shape[1]).permute(0, 2, 1)
        art_source_seqs = torch.cat((art_seqs_estimated, source_seqs), dim=2).to(self.device)
        mel_seqs_repeated = self.nn.synthesizer.nn(art_source_seqs).detach()

        # 4. Generate audio using the vocoder
        audio_seqs_repeated = self.nn.vocoder.resynth(mel_seqs_repeated).detach()

        # 5. Re-extract features
        wav_seqs_len = torch.minimum(audio_len, feat_seqs_len*self.nn.vocoder.frame_size)
        feat_seqs_repeated, feat_seqs_len, feat_seqs_mask = self.nn.feature_extractor.extract_features(audio_seqs_repeated, wav_seqs_len)

        # 6. Compute loss
        min_len = min(feat_seqs.shape[1], feat_seqs_repeated.shape[1])
        feat_seqs = feat_seqs[:, :min_len, :]
        feat_seqs_repeated = feat_seqs_repeated[:, :min_len, :]
        feat_seqs_mask = feat_seqs_mask[:, :min_len]

        inverse_loss = self.losses_fn['mse'](feat_seqs, feat_seqs_repeated, feat_seqs_mask)

        if is_training:
            inverse_loss.backward()
            self.optimizers["inverse_model"].step()

        epoch_record.add("inverse_loss", inverse_loss.item())


    def step_discriminator(self, feat_seqs, seqs_len, seqs_mask,
                           art_seqs, art_seqs_len, art_seqs_mask,
                           epoch_record, regime):
        assert regime in ['train', 'val', 'test']
        is_training = regime == 'train'
        if is_training:
            self.nn.discriminator_model.zero_grad()

        # 1) Real batch training
        if isinstance(self.nn.discriminator_model, LSTM_FF):
            unfolded_art = self.__create_sliding_windows(art_seqs, art_seqs_mask, self.nb_frame_discriminators)
            predicted_labels = self.nn.discriminator_model(unfolded_art)
        else:
            predicted_labels = self.nn.discriminator_model(art_seqs[art_seqs_mask])
        real_labels = torch.full(predicted_labels.shape, 1, dtype=torch.float).to(self.device)
        real_discrimination_loss = self.losses_fn["bce"](predicted_labels, real_labels)
        binarized_predicted_labels = (predicted_labels > .5).float()
        real_accuracy = torch.sum(binarized_predicted_labels == real_labels) / real_labels.shape[0]
        if is_training:
            real_discrimination_loss.backward()

        # 2) Fake batch training
        # a) Generate fake articulatory data
        fake_art_seqs = self.nn.inverse_model(feat_seqs, seqs_len=seqs_len).to(self.device)

        # b) Predict labels
        # We detach to only compute gradients for the discriminator and not the generator
        if isinstance(self.nn.discriminator_model, LSTM_FF):
            unfolded_fake_art = self.__create_sliding_windows(fake_art_seqs, seqs_mask, self.nb_frame_discriminators)
            predicted_labels = self.nn.discriminator_model(unfolded_fake_art.detach())
        else:
            predicted_labels = self.nn.discriminator_model(fake_art_seqs[seqs_mask].detach())

        real_labels = torch.full(predicted_labels.shape, 0, dtype=torch.float).to(self.device)
        fake_discrimination_loss = self.losses_fn["bce"](predicted_labels, real_labels)
        binarized_predicted_labels = (predicted_labels > .5).float()
        fake_accuracy = torch.sum(binarized_predicted_labels == real_labels) / real_labels.shape[0]
        if is_training:
            fake_discrimination_loss.backward()
            self.optimizers["discriminator_model"].step()

        total_loss = (fake_discrimination_loss + real_discrimination_loss) / 2
        total_accuracy = (real_accuracy + fake_accuracy) / 2
        epoch_record.add("discrimination_loss", total_loss.item())
        epoch_record.add("discrimination_accuracy", total_accuracy.item())
        epoch_record.add("discrimination_fake_accuracy", fake_accuracy.item())
        epoch_record.add("discrimination_real_accuracy", real_accuracy.item())

    @staticmethod
    def __create_sliding_windows(art_seqs, art_seqs_mask, nb_frames):
        # 1) create sliding windows
        unfolded_art = art_seqs.unfold(dimension=1, size=nb_frames, step=1)
        # 2) apply mask to avoid considering padded frames
        nb_windows = unfolded_art.shape[1]
        new_mask = art_seqs_mask[:, :nb_windows]
        unfolded_art = unfolded_art[new_mask].transpose(1, 2)
        return unfolded_art
