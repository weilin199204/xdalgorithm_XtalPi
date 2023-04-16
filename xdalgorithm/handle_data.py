import csv
import json
import os

import pandas as pd
from mlflow import (create_experiment, end_run, log_artifacts, log_metric, set_experiment, set_tracking_uri, start_run)
from rdkit import Chem

from rdkit.Chem.Draw.__init__ import MolsToGridImage

def add_mols(mols, mols_per_row=3, legends=None, global_step=None, walltime=None, size_per_mol=(250, 250), pattern=None):
    image=MolsToGridImage(mols,
                               molsPerRow=mols_per_row,
                               subImgSize=size_per_mol,
                               legends=legends,
                               highlightAtomLists=pattern
                               )
    return image

def read_json(path):
    return json.load(open(path))


def read_csv(path):
    return [i for i in csv.reader(open(path))]


def handle_datas(trial_state_path, experiment_state_path,explain_path, exp_name='0',ui_dir="/data/aidd-server/chemal-ui/mlruns"):
    trial_state=read_json(os.path.abspath(trial_state_path))
    experiment_state=read_json(os.path.abspath(experiment_state_path))

    remote_path = ui_dir
    set_tracking_uri("file:" + os.path.abspath(remote_path))

    try:
        create_experiment(exp_name)
    except:
        set_experiment(exp_name)

    ori_arts_dir = "arts_{}_test_state".format(exp_name)
    if not os.path.exists(ori_arts_dir):
        os.mkdir(ori_arts_dir)
    start_run()
    handle_smiles_table(ori_arts_dir, experiment_state['test_state'])
    log_artifacts(ori_arts_dir)
    log_artifacts(explain_path)
    predictor_state = experiment_state['predictor_state']
    worker_state = experiment_state['worker_state']
    handle_predictor_state(predictor_state)
    handle_worker_state(worker_state, "expriment_state", ori_arts_dir)
    log_artifacts(ori_arts_dir)
    end_run()
    for iter_num, run in enumerate(trial_state):
        start_run()
        arts_dir = "arts_{}_{}".format(exp_name,iter_num)
        if not os.path.exists(arts_dir):
            os.mkdir(arts_dir)
        dataset_state = run['dataset_state']
        predictor_state = run['predictor_state']
        worker_state = run['worker_state']
        handle_data_set(dataset_state)
        handle_predictor_state(predictor_state)
        handle_worker_state(worker_state, iter_num, arts_dir)
        log_artifacts(arts_dir)
        end_run()


def handle_smiles_table(arts_dir, data, name="test_state"):
    data['smiles_png'] = [[smi, arts_dir, index] for index, smi in enumerate(data['test_smiles'])]
    df = pd.DataFrame(data)
    df['smile_pngs'] = df['smiles_png'].map(mapping)
    df.to_html("{}/{}.html".format(arts_dir,name), escape=False)
    return


def mapping(i_dir):
    i, arts_dir, index = i_dir
    mol = [Chem.MolFromSmiles(i)]
    image = add_mols(mol, legends=[i])
    f_dir = "{}/smile".format(arts_dir)
    if not os.path.exists(f_dir):
        os.mkdir(f_dir)
    fname = "{}/{}.png".format(f_dir,index)
    use_fname = "smile/{}.png".format(index)
    image.save(fname)
    imgstr = '<img src="{}" /> '.format(use_fname)
    return imgstr


def handle_smiles(arts_dir, smiles_list, indices, from_data, iter_num, inside_idx=0):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    images = add_mols(mols, mols_per_row=10, legends=[str(r) for r in indices])
    image_file = "{}/smiles_{}_iter_{}_{}.jpg".format(arts_dir, from_data, iter_num, inside_idx)
    images.save(image_file)


def handle_data_set(data_set):
    for key in ['aff_labeled_indices', 'aff_unlabeled_indices', 'pose_unlabeled_indices', 'pose_labeled_indices']:
        for d in data_set[key]:
            log_metric("dataset_state." + key, d)
    for key in ['aff_indices', 'aff_data', 'pose_indices', 'pose_label']:
        for d in data_set['new_knowledge'][key]:
            log_metric("dataset_state.new_knowledge." + key, d)


def handle_predictor_state(predictor_state):
    log_metric("iter", predictor_state['iter'])
    for loss in predictor_state['loss_list']:
        log_metric("loss", loss)


def handle_worker_state(worker_state, iter_num, arts_dir):
    acquired_aff_labels = worker_state['acquired_aff_labels']
    acquired_aff_smiles = worker_state['acquired_aff_smiles']

    if acquired_aff_labels and isinstance(acquired_aff_labels[0], list):
        for i, labels in enumerate(acquired_aff_labels):
            smiles = acquired_aff_smiles[i]
            if not smiles:
                continue
            handle_smiles(arts_dir, smiles, labels, 'worker_state', iter_num, inside_idx=i)

    elif acquired_aff_smiles:
        handle_smiles(arts_dir, acquired_aff_smiles, acquired_aff_labels,
                      'worker_state', iter_num)


if __name__=="__main__":
    handle_datas("/mnt/d/fss/trial_states.json", "/mnt/d/fss/experiment_state.json","/mnt/d/fss/explain")
