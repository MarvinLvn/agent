from contextlib import nullcontext
from tqdm import tqdm
import torch

from lib.early_stopping import EarlyStopping
from lib.training_record import TrainingRecord, EpochMetrics
from lib.nn.pad_seqs_frames import pad_seqs_frames


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
        self.sound_scalers = sound_scalers
        self.checkpoint_path = checkpoint_path
        self.device = device

    def train(self):
        training_record = TrainingRecord()
        sound_quantizer_early_stopping = EarlyStopping(
            patience=self.patience, verbose=True, path=self.checkpoint_path
        )
        inverse_model_early_stopping = EarlyStopping(
            patience=self.patience, verbose=True, path=self.checkpoint_path
        )
        is_training_sound_quantizer = True

        for epoch in range(1, self.max_epochs + 1):
            print("== Epoch %s ==" % epoch)

            train_metrics = self.epoch(
                self.train_dataloader,
                is_training=True,
                is_training_sound_quantizer=is_training_sound_quantizer,
            )
            training_record.save_epoch_metrics("train", train_metrics)

            validation_metrics = self.epoch(
                self.validation_dataloader,
                is_training=False,
                is_training_sound_quantizer=False,
            )
            training_record.save_epoch_metrics("validation", validation_metrics)

            if self.test_dataloader is not None:
                test_metrics = self.epoch(
                    self.test_dataloader,
                    is_training=False,
                    is_training_sound_quantizer=False,
                )
                training_record.save_epoch_metrics("test", test_metrics)

            if is_training_sound_quantizer:
                sound_quantizer_early_stopping(
                    validation_metrics.metrics["sound_quantizer_total_loss"],
                    self.nn.sound_quantizer,
                )
            else:
                inverse_model_early_stopping(
                    validation_metrics.metrics["inverse_model_repetition_error"],
                    self.nn,
                )

            if is_training_sound_quantizer:
                if sound_quantizer_early_stopping.early_stop:
                    self.nn.sound_quantizer.load_state_dict(
                        torch.load(self.checkpoint_path)
                    )
                    print("[Sound quantizer] Early stopping")
                    is_training_sound_quantizer = False
            elif inverse_model_early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                print()

        self.nn.load_state_dict(torch.load(self.checkpoint_path))
        return training_record.record

    def epoch(self, dataloader, is_training, is_training_sound_quantizer):
        nb_batch = len(dataloader)
        epoch_record = EpochMetrics(nb_batch)

        if not is_training:
            self.nn.eval()

        with torch.no_grad() if not is_training else nullcontext():
            for batch in tqdm(dataloader, total=nb_batch, leave=False):
                sound_seqs, speaker_seqs, seqs_len, seqs_mask = batch
                sound_seqs = sound_seqs.to("cuda")
                speaker_seqs = speaker_seqs.to("cuda")
                seqs_mask = seqs_mask.to("cuda")

                with torch.no_grad() if not is_training_sound_quantizer else nullcontext():
                    sound_unit_seqs = self.step_sound_quantizer(
                        sound_seqs,
                        speaker_seqs,
                        seqs_mask,
                        epoch_record,
                        is_training=is_training_sound_quantizer,
                    )
                self.step_direct_model(
                    sound_unit_seqs,
                    seqs_len,
                    seqs_mask,
                    epoch_record,
                    is_training=is_training,
                )
                self.step_inverse_model(
                    sound_unit_seqs,
                    seqs_len,
                    seqs_mask,
                    epoch_record,
                    is_training=is_training,
                )

        return epoch_record

    def step_sound_quantizer(
        self, sound_seqs, speaker_seqs, seqs_mask, epoch_record, is_training
    ):
        if is_training:
            self.nn.sound_quantizer.train()
            self.nn.sound_quantizer.requires_grad_(True)

        padded_sound_seqs = pad_seqs_frames(
            sound_seqs, self.nn.sound_quantizer.frame_padding
        )

        if is_training:
            self.optimizers["sound_quantizer"].zero_grad()
        padded_sound_seqs_pred, vq_loss_seqs, quantized_latent_seqs, _, _ = self.nn.sound_quantizer(
            padded_sound_seqs, speaker_seqs, pad_io=False
        )
        total_loss, reconstruction_error, vq_loss = self.losses_fn["vq_vae"](
            padded_sound_seqs_pred, padded_sound_seqs, vq_loss_seqs, seqs_mask
        )
        if is_training:
            total_loss.backward()
            self.optimizers["sound_quantizer"].step()

        epoch_record.add("sound_quantizer_total_loss", total_loss.item())
        epoch_record.add(
            "sound_quantizer_reconstruction_error", reconstruction_error.item()
        )
        epoch_record.add("sound_quantizer_vq_loss", vq_loss.item())

        return quantized_latent_seqs.detach()

    def step_direct_model(
        self, sound_unit_seqs, seqs_len, seqs_mask, epoch_record, is_training
    ):
        if is_training:
            self.nn.inverse_model.eval()
            self.nn.direct_model.train()
            self.nn.direct_model.requires_grad_(True)

        with torch.no_grad():
            art_seqs_estimated = self.nn.inverse_model(
                sound_unit_seqs, seqs_len=seqs_len
            )
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
        self, sound_unit_seqs, seqs_len, seqs_mask, epoch_record, is_training
    ):
        # Inverse model training/evaluation
        # (inverse model estimation → direct model estimation vs. perceived sound)
        if is_training:
            self.nn.inverse_model.train()
            self.nn.direct_model.eval()
            self.nn.direct_model.requires_grad_(False)
            self.nn.sound_quantizer.eval()
            self.nn.sound_quantizer.requires_grad_(False)

            self.optimizers["inverse_model"].zero_grad()

        art_seqs_estimated = self.nn.inverse_model(sound_unit_seqs, seqs_len=seqs_len)
        sound_seqs_estimated = self.nn.direct_model(art_seqs_estimated)
        _, _, _, sound_unit_seqs_estimated = self.nn.sound_quantizer.encode(
            sound_seqs_estimated
        )

        inverse_total, inverse_estimation_error, inverse_jerk = self.losses_fn[
            "inverse_model"
        ](art_seqs_estimated, sound_unit_seqs_estimated, sound_unit_seqs, seqs_mask)
        if is_training:
            inverse_total.backward()
            self.optimizers["inverse_model"].step()

        epoch_record.add(
            "inverse_model_estimation_error", inverse_estimation_error.item()
        )
        epoch_record.add("inverse_model_jerk", inverse_jerk.item())

        # Inverse model repetition error
        # (inverse model estimation → synthesizer → sound quantizer encoder
        # vs. perceived sound → sound quantizer encoder → sound quantizer quantization)
        sound_seqs_produced = self.synthesizer.synthesize_cuda(
            art_seqs_estimated.detach()
        )
        with torch.no_grad():
            _, sound_unit_seqs_produced, _, _ = self.nn.sound_quantizer.encode(
                sound_seqs_produced
            )
        repetition_error = self.losses_fn["mse"](
            sound_unit_seqs_produced, sound_unit_seqs, seqs_mask
        )
        epoch_record.add("inverse_model_repetition_error", repetition_error.item())
