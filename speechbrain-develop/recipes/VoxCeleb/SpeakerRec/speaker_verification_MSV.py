#!/usr/bin/python3
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml

Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
"""
from genericpath import isfile
import os
import sys
import torch
import logging
import torchaudio
sys.path.append("../../../")
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
import pdb

# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
        embeddings = params["mean_var_norm_emb"](
            embeddings, torch.ones(embeddings.shape[0]).to(embeddings.device)
        )
    return embeddings.squeeze(1)

def compute_embedding_loop(data_loader):
    embedding_dict = {}
    embedding_dict_len = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(params["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            spkids = [seg_id.split("--")[0] for seg_id in seg_ids]

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue

            emb = compute_embedding(wavs, lens).unsqueeze(1)
            for i, spk_id in enumerate(spkids):
                embedding_dict.setdefault(spk_id, []).append(emb[i].detach().clone())

    for key, value in embedding_dict.items():
        embedding_dict[key] = torch.cat(value, dim=0).mean(dim=0).unsqueeze(0)

    return embedding_dict

def minmaxscaler(data):
    min_ = torch.min(data, dim=0).values
    max_ = torch.max(data, dim=0).values
    return (data - min_) / (max_ - min_)

def get_submit_file(s, veri_test):
    path = os.path.join(params["output_folder"], trials_name)
    scores = torch.tensor(s)
    scores = minmaxscaler(scores)
    with open(os.path.join(path, "results.csv"), "w") as wf:
        wf.write("utterance_id,speaker_id,score\r\r\n")
        for i, line in enumerate(veri_test):
            enrol_id = line.split(" ")[2].rstrip().strip()
            test_id = line.split(" ")[1].rstrip().split(".")[0].strip()
            wf.write(",".join([test_id + ".wav", enrol_id, str(scores[i].item())]) + "\r\r\n")

    with open(os.path.join(path, "results.txt"), "w") as wf:
        for i, line in enumerate(veri_test):
            lab_pair = line.split(" ")[0].rstrip().split(".")[0].strip()
            enrol_id = line.split(" ")[2].rstrip().strip()
            test_id = line.split(" ")[1].rstrip().split(".")[0].strip()
            wf.write(" ".join([test_id + ".wav", enrol_id, lab_pair, str(scores[i].item())]) + "\n")

def get_verification_scores(veri_test, trials_name=""):
# def get_verification_scores(veri_test):
    """ Computes positive and negative scores given the verification split.
    """
    scores = []
    positive_scores = []
    negative_scores = []

    save_folder = params["output_folder"]
    save_dir = os.path.join(save_folder, trials_name)
    save_file = os.path.join(save_folder, trials_name, "scores.txt")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    s_file = open(save_file, "w")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # creating cohort for score normalization
    
    for i, line in enumerate(veri_test):

        # Reading verification file (enrol_file test_file label)
        lab_pair = int(line.split(" ")[0].rstrip().split(".")[0].strip())

        enrol_id = line.split(" ")[2].rstrip().strip()
        test_id = line.split(" ")[1].rstrip().split(".")[0].strip()

        enrol = enrol_dict[enrol_id]
        test = test_dict[test_id]

        # Compute the score for the given sentence
        score = similarity(enrol, test)[0]

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score))
        scores.append(score)

        if lab_pair == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    s_file.close()

    get_submit_file(scores, veri_test)

    return positive_scores, negative_scores


def dataio_prep(params):
    "Creates the dataloaders and their data processing pipelines."

    data_folder = params["data_folder"]

    # 1. Declarations:

    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data"], replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [enrol_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # 4 Create dataloaders
    enrol_dataloader = sb.dataio.dataloader.make_dataloader(
        enrol_data, **params["enrol_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )

    return enrol_dataloader, test_dataloader

if __name__ == "__main__":
    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_paths = params["verification_file"]

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # here we create the datasets objects as well as tokenization and encoding
    enrol_dataloader, test_dataloader = dataio_prep(params)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(params["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])

    # Computing  enrollment and test embeddings
    logger.info("Computing enroll/test embeddings...")

    # First run
    enrol_dict = compute_embedding_loop(enrol_dataloader)
    test_dict = compute_embedding_loop(test_dataloader)

    # Second run (normalization stats are more stable)
    # enrol_dict = compute_embedding_loop(enrol_dataloader)
    # test_dict = compute_embedding_loop(test_dataloader)

    # if "score_norm" in params:
    #     train_dict = compute_embedding_loop(train_dataloader)
    
    # Compute the EER
    logger.info("Computing EER..")

    for veri_file_path in veri_file_paths:
        with open(veri_file_path) as f:
            veri_test = [line.rstrip() for line in f]
            trials_name = veri_file_path.split("/")[-1]
            logger.info("Computing EER on " + trials_name)
            if len(veri_file_paths) == 1:
                trials_name = ""
            positive_scores, negative_scores = get_verification_scores(veri_test, trials_name)

    del enrol_dict, test_dict

    logger.info("get score finished!!!!!")
