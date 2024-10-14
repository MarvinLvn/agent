### Installation

```sh
conda create --name agent python=3.8 && conda activate agent

# 1. Install hifi-gan dependencies
cd hifi-gan
pip install -e .

# 2. Install dependencies
cd ..
conda env update -f env.yml 
```

```sh
# Install this repo
git clone https://github.com/MarvinLvn/agent
cd agent
conda create --name agent python=3.8 && conda activate agent
pip install -r requirements.txt

# Install hifi-gan dependency
git clone https://github.com/MarvinLvn/hifi-gan
cd hifi-gan
pip install -e .
```

### Datasets

1) Get pb2009 to train the synthesizer, evaluate the agent, and fine-tune the vocoder:

```sh
Give path in gpu3
```
2) Get the audiocite corpus prepared (french audiobooks) used to train the agent:
-
```sh
Give path in gpu3
```

3) (Optional) If you need to retrain the vocoder, get this [multi-speaker/multi-lingual dataset](https://huggingface.co/datasets/mbarnig/lb-de-fr-en-pt-12800-TTS-CORPUS)


### Preprocess datasets

Resample wav files, extract mel, extract articulatory parameters and preprocess labels(relevant):

```sh
python preprocess_datasets.py
```

This script will preprocess datasets contained in `datasets_infos.yaml`.

## Create ABX

for speaker in *; do for file in $speaker/*_utt.csv; do new_file=$speaker/$(basename ${file/_utt.csv/}).lab; tail -n +2 $file | cut -f1 | sed 's/[,;:.!?]//g' | tr '\n' ' ' | sed 's/^ *//; s/ *$//' > $new_file; done; done

Install 
```sh
conda create -n mfa -c conda-forge montreal-forced-aligner
mfa model download acoustic french_mfa
mfa model download dictionary french_mfa
mfa model download g2p french_mfa
```

Add missing pronunciations found in heldout and test into the dictionary

```sh
mfa g2p /home/engaclew/agent/datasets/heldout/MFA french_mfa /home/engaclew/agent/datasets/heldout/g2pped_oovs.txt --dictionary french_mfa
mfa g2p /home/engaclew/agent/datasets/test/MFA french_mfa /home/engaclew/agent/datasets/test/g2pped_oovs.txt --dictionary french_mfa
```

If these pronunciations seem sensible, then you can add them to the dictionary:

```sh
mfa model add_words french_mfa /home/engaclew/agent/datasets/heldout/g2pped_oovs.txt
mfa model add_words french_mfa /home/engaclew/agent/datasets/test/g2pped_oovs.txt
```

Now we can align:

```
mfa align /home/engaclew/agent/datasets/heldout/MFA french_mfa french_mfa /home/engaclew/agent/datasets/heldout/MFA_aligned
```

# Superb

Probing on the phone recognition and speaker identification tasks were done using the [superb benchmark](https://github.com/s3prl/s3prl/tree/main).
Instructions for data downloading/preparation can be found [here](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md#sid-speaker-identification).

1) Install
```sh
conda create --name superb python=3.8
conda activate superb
git clone https://github.com/s3prl/s3prl.git
cd s3prl
pip install -e .
pip install transformers==4.28.0 tensorboardX editdistance joblib
```

2) Train (phone recognition):

```sh
python run_downstream.py -m train -u hf_wav2vec2_custom -d ctc -k facebook/wav2vec2-base-10k-voxpopuli -c downstream/ctc/libriphone.yaml -n w2v_PR_layer_0 -s hidden_states -l 0
```

3) Test (phone recognition)

```sh
python run_downstream.py -m evaluate -e result/downstream/w2v_PR_layer_0/dev-best.ckpt
```

4) Train (speaker identification)

```sh
python run_downstream.py -m train -u hf_wav2vec2_custom -d voxceleb1 -k facebook/wav2vec2-base-10k-voxpopuli -n w2v_SID_layer_0 -s hidden_states -l 0
```

5) Test (speaker identification)

```sh
python run_downstream.py -m evaluate -e result/downstream/w2v_SID_layer_0/dev-best.ckpt
```