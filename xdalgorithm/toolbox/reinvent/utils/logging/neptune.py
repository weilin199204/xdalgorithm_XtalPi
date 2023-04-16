import io
import PIL
import umap
import numpy as np
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

def add_mols(neptune, tag, mols, 
            mols_per_row=1, 
            legends=None, 
            global_step=None, 
            size_per_mol=(300, 300),
            pattern=None):
    """
    Adds molecules in a grid.
    """
    try:
        image = MolsToGridImage(mols, 
                    molsPerRow=mols_per_row, 
                    subImgSize=size_per_mol, 
                    legends=legends,
                    highlightAtomLists=pattern)
        # convert IPython.display.image to PIL Image object
        buf = io.BytesIO(image.data)
        pilimg = PIL.Image.open(buf)
    except PIL.UnidentifiedImageError:
        pilimg = PIL.Image.new(
                        'RGBA', 
                        (size_per_mol[0] * mols_per_row, size_per_mol[1]), 
                        (200, 100, 100))

    add_image(neptune, tag, pilimg, global_step)

def add_image(neptune, tag, image, global_step=None):
    """
    Adds an image from a PIL image.
    """
    neptune.log_image(tag, x=global_step, y=image, image_name="step {}".format(global_step))

def add_frequent_scaffolds(neptune, tag, mols, top_frequent, 
                        mols_per_row=1,
                        global_step=None,
                        size_per_mol=(300, 300),
                        pattern=None):
    scaffold_dict = dict()
    for mol in mols:
        try:
            current_scaffold = GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(current_scaffold)
        except:
            continue
        if scaffold_smiles == "":
            continue
        if scaffold_smiles in scaffold_dict.keys():
            scaffold_dict[scaffold_smiles] += 1
        else:
            scaffold_dict[scaffold_smiles] = 1
    total_num = len(mols)
    scaffold_list = list(scaffold_dict.keys())
    scaffold_list = sorted(scaffold_list, key=lambda x: scaffold_dict[x], reverse=True)
    if len(scaffold_list) <= top_frequent:
        picked_scaffold_list = scaffold_list
    else:
        picked_scaffold_list = scaffold_list[:top_frequent]
    scaffold_ratio = ["%.2f%%"%(100 * scaffold_dict[scaffold] / total_num) for scaffold in picked_scaffold_list]
    # return picked_scaffold_list, scaffold_ratio
    picked_scaffold_mols = [Chem.MolFromSmiles(i) for i in picked_scaffold_list]

    try:
        image = MolsToGridImage(picked_scaffold_mols, 
                                molsPerRow=mols_per_row,
                                subImgSize=size_per_mol,
                                legends=scaffold_ratio,
                                highlightAtomLists=pattern)
        # convert IPython.display.image to PIL Image object
        buf = io.BytesIO(image.data)
        pilimg = PIL.Image.open(buf)
    except PIL.UnidentifiedImageError:
        pilimg = PIL.Image.new('RGBA', (size_per_mol[0] * mols_per_row, size_per_mol[1]), (200, 100, 100))

    add_image(neptune, tag, pilimg, global_step)

def get_fingerprint(mol):
    morganfp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False,nBits=512).ToBitString()
    fp_array = np.array([int(i) for i in morganfp])
    return fp_array


def get_2D_umap(features, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(features)
    return u

def draw_umap(neptune, tag, mols_groups, legends, global_step=None):
    group_fp = []
    group_label = []

    for idx, each_group in enumerate(mols_groups):
        for mol in each_group:
            fp_array = get_fingerprint(mol)
            if fp_array is not None:
                group_fp.append(fp_array)
                group_label.append(idx)
    feature_table = np.array(group_fp)
    umap_feature = get_2D_umap(feature_table, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    group_array = np.array(group_label)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    colors = ['b', 'y', 'c', 'm', 'r']
    for idx, _ in enumerate(mols_groups):
        ax.scatter(umap_feature[group_array == idx, 0], umap_feature[group_array == idx, 1],
                   c=colors[idx], marker='o', alpha=0.6, label=legends[idx])
    if legends is not None:
        plt.legend()

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = PIL.Image.open(buf)

    add_image(neptune, tag, image, global_step)