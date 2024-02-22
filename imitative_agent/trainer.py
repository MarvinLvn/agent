from contextlib import nullcontext
import torch
from tqdm import tqdm

from lib.early_stopping import EarlyStopping
from lib.training_record import TrainingRecord, EpochMetrics

from lib.nn.pad_seqs_frames import pad_seqs_frames
from lib.nn.simple_lstm import LSTM_FF
from lib.nn.feedforward import FeedForward


class Trainer:
    def __init__(
        self,
        nn,
        optimizers,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        losses_fn,
        max_epochs,
        patience,
        synthesizer,
        sound_scalers,
        checkpoint_path,
        nb_frames_discriminator=1,
        device="cuda",
    ):
        self.nn = nn.to(device)
        self.optimizers = optimizers
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.losses_fn = losses_fn
        self.max_epochs = max_epochs
        self.patience = patience
        self.synthesizer = synthesizer
        self.train_artloader, self.validation_artloader, self.test_artloader = None, None, None
        if self.nn.discriminator_model is not None:
            self.train_artloader, self.validation_artloader, self.test_artloader = synthesizer.get_dataloaders()
        self.sound_scalers = sound_scalers
        self.checkpoint_path = checkpoint_path
        self.nb_frame_discriminators = nb_frames_discriminator
        self.device = device

    def train(self):
        training_record = TrainingRecord()
        self.train_model_part(
            training_record, self.epoch_inverse_model, "inverse_model_repetition_error"
        )
        return training_record.record

    def train_model_part(self, training_record, epoch_fn, early_stopping_metric):
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=True, path=self.checkpoint_path
        )

        for epoch in range(1, self.max_epochs + 1):
            print("== Epoch %s ==" % epoch)

            train_metrics = epoch_fn(self.train_dataloader, self.train_artloader, is_training=True)
            training_record.save_epoch_metrics("train", train_metrics)

            validation_metrics = epoch_fn(self.validation_dataloader, self.validation_artloader, is_training=False)
            training_record.save_epoch_metrics("validation", validation_metrics)

            if self.test_dataloader is not None:
                test_metrics = epoch_fn(self.test_dataloader, self.test_artloader, is_training=False)
                training_record.save_epoch_metrics("test", test_metrics)

            early_stopping(validation_metrics.metrics[early_stopping_metric], self.nn)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                print()

        self.nn.load_state_dict(torch.load(self.checkpoint_path))

    def epoch_inverse_model(self, dataloader, art_loader, is_training):
        nb_batch = len(dataloader)
        epoch_record = EpochMetrics(nb_batch)
        if art_loader is not None:
            art_loader_discriminator = iter(art_loader)

        if not is_training:
            self.nn.eval()

        with torch.no_grad() if not is_training else nullcontext():
            for batch in tqdm(dataloader, total=nb_batch, leave=False):
                sound_seqs, seqs_len, seqs_mask = batch
                sound_seqs = sound_seqs.to("cuda")
                seqs_mask = seqs_mask.to("cuda")

                self.step_direct_model(
                    sound_seqs,
                    seqs_len,
                    seqs_mask,
                    epoch_record,
                    is_training=is_training,
                )

                self.step_inverse_model(
                    sound_seqs,
                    seqs_len,
                    seqs_mask,
                    epoch_record,
                    is_training=is_training,
                )

                if self.nn.discriminator_model is not None and art_loader is not None:
                    try:
                        # We sample a batch of real articulatory trajectories
                        art_seqs, speaker_art_seqs, art_seqs_len, art_seqs_mask = next(art_loader_discriminator)
                        art_seqs = art_seqs.to(self.device)
                    except StopIteration:
                        # Recreate iterator when the whole data has been consumed
                        art_loader_discriminator = iter(art_loader)
                        art_seqs, speaker_art_seqs, art_seqs_len, art_seqs_mask = next(art_loader_discriminator)
                        art_seqs = art_seqs.to(self.device)
                    self.step_discriminator(sound_seqs,
                                            # Input to the generator from which to generate articulatory data
                                            seqs_len,
                                            seqs_mask,
                                            art_seqs,  # Real articulatory data
                                            art_seqs_len,
                                            art_seqs_mask,
                                            epoch_record=epoch_record,
                                            is_training=is_training)
        return epoch_record

    def step_direct_model(
        self, sound_seqs, seqs_len, seqs_mask, epoch_record, is_training
    ):
        if is_training:
            self.nn.inverse_model.eval()
            self.nn.direct_model.train()
            self.nn.direct_model.requires_grad_(True)

        with torch.no_grad():
            art_seqs_estimated = self.nn.inverse_model(sound_seqs, seqs_len=seqs_len)
        sound_seqs_produced = self.synthesizer.synthesize_cuda(art_seqs_estimated)
        sound_seqs_produced = self.sound_scalers["synthesizer"].inverse_transform(
            sound_seqs_produced
        )
        sound_seqs_produced = self.sound_scalers["agent"].transform(sound_seqs_produced)

        if is_training:
            self.optimizers["direct_model"].zero_grad()
        sound_seqs_estimated = self.nn.direct_model(art_seqs_estimated)
        direct_model_loss = self.losses_fn["mse"](
            sound_seqs_estimated, sound_seqs_produced, seqs_mask
        )
        if is_training:
            direct_model_loss.backward()
            self.optimizers["direct_model"].step()

        epoch_record.add("direct_model_estimation_error", direct_model_loss.item())

    def step_inverse_model(
        self, sound_seqs, seqs_len, seqs_mask, epoch_record, is_training
    ):
        # Inverse model training/evaluation
        # (inverse model estimation → direct model estimation vs. perceived sound)
        if is_training:
            self.nn.inverse_model.train()
            self.nn.direct_model.eval()
            self.nn.direct_model.requires_grad_(False)
            self.optimizers["inverse_model"].zero_grad()

        art_seqs_estimated = self.nn.inverse_model(sound_seqs, seqs_len=seqs_len)
        sound_seqs_estimated = self.nn.direct_model(art_seqs_estimated)

        if isinstance(self.nn.discriminator_model, LSTM_FF):
            unfolded_art = self.__create_sliding_windows(art_seqs_estimated, seqs_mask, self.nb_frame_discriminators)
            predicted_labels = self.nn.discriminator_model(unfolded_art)
        else:
            predicted_labels = self.nn.discriminator_model(art_seqs_estimated[seqs_mask])

        inverse_total, inverse_estimation_error, inverse_jerk, fool_discrimination_loss = self.losses_fn[
            "inverse_model"
        ](art_seqs_estimated, sound_seqs_estimated, sound_seqs, seqs_mask, predicted_labels)
        if is_training:
            inverse_total.backward()
            self.optimizers["inverse_model"].step()

        epoch_record.add(
            "inverse_model_estimation_error", inverse_estimation_error.item()
        )
        epoch_record.add("inverse_model_jerk", inverse_jerk.item())
        epoch_record.add("fool_discrimination_loss", fool_discrimination_loss.item())

        # Inverse model repetition error
        # (inverse model estimation → synthesizer vs. perceived sound)
        sound_seqs_produced = self.synthesizer.synthesize_cuda(
            art_seqs_estimated.detach()
        )
        repetition_error = self.losses_fn["mse"](
            sound_seqs_produced, sound_seqs, seqs_mask
        )
        epoch_record.add("inverse_model_repetition_error", repetition_error.item())

    def step_discriminator(self, sound_unit_seqs, seqs_len, seqs_mask,
                           art_seqs, art_seqs_len, art_seqs_mask,
                           epoch_record, is_training):
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
        fake_art_seqs = self.nn.inverse_model(sound_unit_seqs, seqs_len=seqs_len).to(self.device)

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
