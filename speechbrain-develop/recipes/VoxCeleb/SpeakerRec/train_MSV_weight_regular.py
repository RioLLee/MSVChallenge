#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import pdb
import sys
import random
import torch
import torchaudio
import sys
sys.path.append("../../../")
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

# from speechbrain.lobes.mmd import mmd

# from speechbrain.nnet.DELTA import reg_fea_map, get_register_hook, remove_hooks

import logging

class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        # wavs, lens = batch.sig2

        # teacher_wavs, teacher_lens = batch.sig1
        wavs, lens = batch.sig

        if stage == sb.Stage.TRAIN:

            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                wavs_aug = augment(wavs, lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        # if stage == sb.Stage.TRAIN:
        #     teacher_feats = self.modules.compute_features(teacher_wavs)

        # feats fbank random mask
        # feats = specAug(feats)
        # feats.size() [B*6,T,F]

        # Spec
        spec_prob = torch.rand(1)
        if spec_prob > 0.7:
            feats = self.modules.specAugment(feats)
            # if stage == sb.Stage.TRAIN:
            #     teacher_feats = self.modules.specAugment(teacher_feats)

        feats = self.modules.mean_var_norm(feats, lens)
        # if stage == sb.Stage.TRAIN:
        #     teacher_feats = self.modules.mean_var_norm(teacher_feats, teacher_lens)

        # Embeddings + speaker classifier
        # embeddings = self.modules.student_model(feats, lens)
        # if stage == sb.Stage.TRAIN:
        #     teacher_embeddings = self.modules.teacher_model(feats, lens)

        embeddings = self.modules.embedding_model(feats, lens)

        # if stage == sb.Stage.TRAIN:
        #     teacher_embeddings = self.modules.teacher_model(teacher_feats, teacher_lens)
        
        #B*6,1,192----->B,6,192,192
        outputs = self.modules.classifier(embeddings)

        # if stage == sb.Stage.TRAIN:
        #     domain_loss = mmd(teacher_embeddings, embeddings)
        # else:
        #     domain_loss = torch.tensor(0.).to(self.device)

        '''
        spkid, _ = batch.spk_id_encoded
        
        if stage == sb.Stage.TRAIN:
            teacher_embeddings = torch.cat([teacher_embeddings] * self.n_augment)
            spkid = torch.cat([spkid] * self.n_augment, dim=0)

        teacher_dict = {}
        for i, spk_id in enumerate(spkid):
            teacher_dict.setdefault(spk_id.item(), []).append(teacher_embeddings[i])

        centric_teacher_dict = {}
        for key, embeds in teacher_dict.items():
            centric_teacher_dict[key] = torch.cat(embeds, dim=0).mean(dim=0).unsqueeze(0)

        teachers = []
        for spk_id in spkid:
            teachers.append(centric_teacher_dict[spk_id.item()])

        centric_teacher_embeddings = torch.cat(teachers, dim=0).to(self.device)

        domain_loss = mmd(centric_teacher_embeddings, embeddings.squeeze(1))
        '''

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        weights0 = {}
        weights_now  = {}
        weight_reg = 0.0

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            spkid = torch.cat([spkid] * self.n_augment, dim=0)

            for name, params in self.modules.teacher_model.named_parameters():
                if 'norm' not in name:
                    weights0[name] = params

            for name, params in  self.modules.embedding_model.named_parameters():
                if 'norm' not in name:
                    weights_now[name] = params
                    weight_reg += 5 * (torch.pow(params - weights0[name], 2).sum() / 2)

            # fea_loss = reg_fea_map(self.device, self.n_augment)
        # else:
        #     fea_loss = torch.tensor(0.).to(self.device)

        loss = self.hparams.compute_cost(predictions, spkid, lens)
        logging.debug(f"loss is ---------> {loss}")

        loss = loss + self.hparams.domain_loss_rate * weight_reg
        logging.debug(f"domain_loss is ---------> {weight_reg}") 

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

        for p in self.modules.teacher_model.parameters():
            p.requires_grad = False

        # if stage == sb.Stage.TRAIN:
        #     get_register_hook(self.modules.teacher_model, self.modules.student_model)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # if stage == sb.Stage.TRAIN:
        #     remove_hooks()

        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
                num_to_keep=3
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    # @sb.utils.data_pipeline.takes("wav1", "wav2", "start1", "stop1", "duration1", "start2", "stop2", "duration2")
    # @sb.utils.data_pipeline.provides("sig1", "sig2")
    # def audio_pipeline(wav1, wav2, start1, stop1, duration1, start2, stop2, duration2):
    #     start1 = int(start1)
    #     stop1 = int(stop1)

    #     start2 = int(start2)
    #     stop2 = int(stop2)

    #     num_frames1 = stop1 - start1 if stop1 - start1 < snt_len_sample else snt_len_sample
    #     num_frames2 = stop2 - start2 if stop2 - start2 < snt_len_sample else snt_len_sample

    #     sig1, fs = torchaudio.load(
    #         wav1, num_frames=num_frames1, frame_offset=start1
    #     )

    #     sig2, fs = torchaudio.load(
    #         wav2, num_frames=num_frames2, frame_offset=start2
    #     )
 
    #     sig1 = sig1.transpose(0, 1).squeeze(1)
    #     sig2 = sig2.transpose(0, 1).squeeze(1)

    #     return sig1, sig2    # sig1 means iphone0.25m, sig2 means all
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            if (duration_sample - snt_len_sample - 1)  < 0:
                start = 0
            else:
                start = random.randint(0, duration_sample - snt_len_sample - 1)

            stop = start + snt_len_sample if start + snt_len_sample < int(stop) else int(stop)
        else:
            start = int(start)
            stop = int(stop)

        num_frames = stop - start if stop - start < snt_len_sample else snt_len_sample
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    # lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    lab_enc_file = hparams["label_encoder"]
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])
    # sb.dataio.dataset.set_output_keys(datasets, ["id", "sig1", "sig2", "spk_id_encoded"])

    return train_data, valid_data, label_encoder


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    # veri_file_path = os.path.join(
    #     hparams["save_folder"], os.path.basename(hparams["verification_file"])
    # )
    # download_file(hparams["verification_file"], veri_file_path)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)


    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
