import os
import base64
from io import BytesIO
import ipywidgets
from IPython.core.display import display

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from .base_parameter_helper import BaseParameterHelper


DEFAULT_PHARMACOPHORE = ["Donor", "Acceptor","Aromatic", "Hydrophobe", "LumpedHydrophobe"]

class P4coreAlignParameterHelper(BaseParameterHelper):
    """
    A 3D Pharmacophore Model RL parameters constructor.

    Template:

        ```
        {
            "component_type": "pharmacophore_align",
            "name": "pharmacophore_align",
            "weight": 4,
            "model_path": None,
            "smiles": [],
            "specific_parameters": {
                "template_mol_file": "lig.sdf",
                "d_upper": 1.5,
                "d_lower": 1.5,
                "keep": ["Donor", "Acceptor","Aromatic", "Hydrophobe", "LumpedHydrophobe"],
                "pharmacophore_idxs": [0,1,4,13],
                "stepwise_reward_weights": [0.01,0.1,0.4,2.0],
                "conformers_num":5,
                "pList_max_allowed":30,
                "max_to_try":20,
                "num_desired": 5,
                "penalty_weight":3,
                "lig_name":"6S1",
                "pdb_name":"5KCV.pdb",
                "atom_width":3
            }
        }
        ```

    Usage:

        ```
        sdfile = '2JKM_BII.sdf'
        pdfile = '2JKM.pdb'
        lig_name = 'BII'
        availableP4coreModels = ["Donor", "Acceptor","Aromatic", "Hydrophobe", "LumpedHydrophobe"]
        phar3dhelper = Pharm3DParameterHelper(
            pdfile, 
            sdfile, 
            lig_name=lig_name, 
            availableP4coreModels=availableP4coreModels, 
            space_penalty=True)

        # show interactive ipywidget table
        phar3dhelper.showInteractiveTable()

        # generate reinforcement learning parameters in JSON
        phar3dhelper.genRLJSON()

        ```

    """
    JSON_TEMPLATE = {
        "component_type": "pharmacophore_align",
        "name": "pharmacophore_align",
        "weight": 4,
        "model_path": None,
        "smiles": [],
        "specific_parameters": {
            "template_mol_file": "lig.sdf",
            "d_upper": 1.5,
            "d_lower": 1.5,
            "keep": ["Donor", "Acceptor","Aromatic", "Hydrophobe", "LumpedHydrophobe"],
            "pharmacophore_idxs": [0,1,4,13],
            "stepwise_reward_weights": [0.01,0.1,0.4,2.0],
            "conformers_num":5,
            "pList_max_allowed":30,
            "max_to_try":20,
            "num_desired": 5,
            "penalty_weight":3,
            "lig_name":"6S1",
            "pdb_name":"5KCV.pdb",
            "atom_width":3
        }
    }

    def __init__(self, pdb, sdf, pharmacophore_idxs=None, 
                lig_name='MOL',
                availableP4coreModels=None, 
                weight=4,
                d_upper=1.5,
                d_lower=1.5,
                pharmacophore_weights=None,
                stepwise_reward_weights=[0.1, 0.25, 1.0, 2.0],
                pList_max_allowed=30,
                max_to_try=20,
                num_desired=5,
                atom_width=1.5,
                space_penalty=3):
        """
        Args:
        - component_type: 定义一个药效团的打分函数,具体打分原理见附录
        - name: 用户自定义这个component的名字 
        - weight: 这个component的权重，供计算综合得分用
        - model_path: model_path:这个component不使用模型，为null
        - smiles: smiles:这个component不使用smiles,为空列表
        - sdf: template_mol_file：参照分子的结合构象的sdf文件
        - d_upper: 生成分子的药效团pairwise distance允许大于参照分子中相应距离的值
        - d_lower: 生成分子的药效团pairwise distance允许低于参照分子中相应距离的值
        - keep: 需要保留的药效团类型
        - pharmacophore_idxs: 需要保留的药效团的索引值（见附录）
        - pharmacophore_weights: 药效团的奖励权重设置 (默认都为1.0)
        - stepwise_reward_weights: 药效团模型阶梯式打分中每步的权重
        - pList_max_allowed: 当GetAllPharmacophoreMatches得到多个匹配的组合，
        当组合的数量多于pList_max_allowed时，随机选取其中的 pList_max_allowed个pList
        - max_to_try: EmbedPharmacophore允许失败的次数
        - num_desired: EmbedPharmacophore产生的期望构象数
        - atom_width: 每个原子的半径，单位埃
        - lig_name: 利用复合物结合来得到生成分子不该进入的排除体积
        - penalty_weigth: 进入排斥体积后的惩罚

        """
        # return a selected_pharms_index
        self.selected_pharms_index = set()
        self.sdf = sdf
        self.pdb = pdb
        self.pharmacophore_idxs = pharmacophore_idxs
        self.lig_name = lig_name
        self.availableP4coreModels = availableP4coreModels
        self.weight = weight
        self.d_upper = d_upper
        self.d_lower = d_lower
        self.pharmacophore_weights = pharmacophore_weights
        self.stepwise_reward_weights = stepwise_reward_weights
        self.pList_max_allowed = pList_max_allowed
        self.max_to_try = max_to_try
        self.num_desired = num_desired
        self.atom_width = atom_width
        self.space_penalty = space_penalty

        if not os.path.exists(self.sdf):
            raise Exception("cannot find sd file {}".format(self.sdf))
        
        if not os.path.exists(self.pdb):
            raise Exception("cannot find pdb file {}".format(self.pdb))
            
        if not self.availableP4coreModels:
            self.availableP4coreModels = DEFAULT_PHARMACOPHORE
        
        self.rl_template_dict = dict({
            "component_type": "pharmacophore_align", 
            "name": "pharmacophore_align", 
            "weight": self.weight, 
            "model_path": None, 
            "smiles": [],
            "specific_parameters": {
                "template_mol_file": self.sdf,
                "d_upper": self.d_upper,
                "d_lower": self.d_lower, 
                "keep": self.availableP4coreModels, 
                "pharmacophore_idxs": self.pharmacophore_idxs, 
                "pharmacophore_weights": self.pharmacophore_weights,
                "stepwise_reward_weights": self.stepwise_reward_weights,
                "pList_max_allowed": self.pList_max_allowed,
                "max_to_try": self.max_to_try,
                "num_desired": self.num_desired,
                "atom_width": self.atom_width,  
                "lig_name": self.lig_name,  
                "pdb_name": self.pdb,
                "penalty_weight": self.space_penalty if self.space_penalty else 3
            }
        })

    def _make_clickable(self, val):
        print(val)
        return '<a href="{}">{}</a>'.format(val,val)

    def _image_base64(self, im):
        with BytesIO() as buffer:
            im.save(buffer, 'png')
            return base64.b64encode(buffer.getvalue()).decode()

    def _image_formatter(self, im):
        return f'<img src="data:image/png;base64,{self._image_base64(im)}">'

    def _checkbox_changed(self, b):
        if not b['old'] and b['new']:
            self.selected_pharms_index.add(b['owner'].description)
            
        if b['old'] and not b['new']:
            self.selected_pharms_index.remove(b['owner'].description)
    
    def _show_atom_number(self, mol, label):
        for atom in mol.GetAtoms():
            atom.SetProp(label, str(atom.GetIdx()))
        return mol

    def reload_template(self):
        """
        Reload the JSON template from in-class attributes.
        """
        self.rl_template_dict = dict({
            "component_type": "pharmacophore_align", 
            "name": "pharmacophore_align", 
            "weight": self.weight, 
            "model_path": None, 
            "smiles": [],
            "specific_parameters": {
                "template_mol_file": self.sdf,
                "d_upper": self.d_upper,
                "d_lower": self.d_lower, 
                "keep": self.availableP4coreModels, 
                "pharmacophore_idxs": self.pharmacophore_idxs, 
                "pharmacophore_weights": self.pharmacophore_weights,
                "stepwise_reward_weights": self.stepwise_reward_weights,
                "pList_max_allowed": self.pList_max_allowed,
                "max_to_try":self.max_to_try,
                "num_desired":self.num_desired,
                "atom_width": self.atom_width,  
                "lig_name": self.lig_name,  
                "pdb_name": self.pdb,
                "penalty_weight": self.space_penalty if self.space_penalty else 3
            }
        })
    
    def generate_template(self):
        """
        Show a interactive window to select pharmacophore_idxs and return the JSON.
        """
        if not self.pharmacophore_idxs:
            self.rl_template_dict['specific_parameters']['pharmacophore_idxs'] = [int(i) for i in list(self.selected_pharms_index)]
            self.rl_template_dict['specific_parameters']['pharmacophore_weights'] = [1.0] * len(self.rl_template_dict['specific_parameters']['pharmacophore_idxs'])
        else:
            self.rl_template_dict['specific_parameters']['pharmacophore_idxs'] = self.pharmacophore_idxs
            self.rl_template_dict['specific_parameters']['pharmacophore_weights'] = [1.0] * len(self.rl_template_dict['specific_parameters']['pharmacophore_idxs'])
        
        return self.rl_template_dict
        
    def show_interactive_table(self, pdb=None, sdf=None, availableP4coreModels=None, predefinedBaseFeatureFile=None):
        """
        Show a interactive window to select pharmacophore_idxs and return the JSON.

        Args:
        - pdb: input complex structure.
        - sdf: input ligand sd file.
        - availableP4coreModels: the list of available pharmacophore model to be included.

        """
        if not sdf:
            sdf = self.sdf
        if not os.path.exists(sdf):
            raise Exception("cannot find sd file {}".format(sdf))

        if not pdb:
            pdb = self.pdb
        if not os.path.exists(pdb):
            raise Exception("cannot find pdb file {}".format(pdb))
            
        if not availableP4coreModels:
            availableP4coreModels = self.availableP4coreModels

        if predefinedBaseFeatureFile:
            if os.path.exists(predefinedBaseFeatureFile):
                predefinedBaseFeatFilepath = predefinedBaseFeatureFile
            else:
                raise Exception("Pre-defined feature file {} does not exist".format(predefinedBaseFeatureFile))
        else:
            import xdalgorithm
            predefinedBaseFeatFilepath = os.path.join(
                os.path.dirname(xdalgorithm.__file__), 
                "toolbox/interaction_fingerprints/data/BaseFeatures.fdef")

        fdef = AllChem.BuildFeatureFactory(predefinedBaseFeatFilepath)
        
        ligand_in_complex = [m for m in Chem.SDMolSupplier(self.sdf)][0]
        feats = fdef.GetFeaturesForMol(ligand_in_complex)
        available_feats = [f for f in feats if f.GetFamily() in availableP4coreModels]
        available_feats_pts = [list(x.GetPos()) for x in available_feats]

        # annotate atoms with index number
        ligand_with_atom_ids = self._show_atom_number(ligand_in_complex, "molAtomMapNumber")
        rdDepictor.Compute2DCoords(ligand_with_atom_ids)

        chk_objs = []
        pharm_models = pd.DataFrame(columns=['Index', 'Feature', 'Type', 'Atom ID'])
        for i, feat in enumerate(available_feats):
            atomids = feat.GetAtomIds()
            pharm_mol_img = Draw.MolToImage(ligand_in_complex, 
                                    highlightAtoms=atomids,
                                    highlightColor=(0,1,0), 
                                    useSVG=True)
            encoded_img = self._image_formatter(pharm_mol_img)
            
            chk_obj = ipywidgets.Checkbox(
                value=False,
                description='{}'.format(i),
                disabled=False,
                indent=False
            )
            chk_obj.observe(self._checkbox_changed, "value")
            chk_objs.append(chk_obj)
            
            pharm_models = pharm_models.append({
                'Index': i,
                'Feature': feat.GetFamily(),
                'Type': feat.GetType(),
                'Atom ID': list(atomids),
            }, ignore_index=True)
        
        mol_with_pharm_imgs = []
        for pharm_idx in range(pharm_models['Atom ID'].size):
            mol_with_pharm_img = Draw.MolToImage(ligand_with_atom_ids, 
                                    size=(500, 500),
                                    highlightAtoms=pharm_models['Atom ID'][pharm_idx],
                                    highlightColor=(0,1,0), 
                                    useSVG=True)
            with BytesIO() as buffer:
                mol_with_pharm_img.save(buffer, 'png')
                pharm_img_bytes = buffer.getvalue()
                mol_with_pharm_imgs.append(pharm_img_bytes)
        
        # init slider linked image viewer for pharmacophore model
        pharm_viewer = ipywidgets.Image(
            value=mol_with_pharm_imgs[0],
            format='png',
            width=400,
            height=400,
        )

        pharm_slider = ipywidgets.IntSlider(
            value=0,
            min=0,
            max=pharm_models['Atom ID'].size-1,
            step=1,
            description='Index:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )

        def on_slider_change(change):
            pharm_viewer.value = mol_with_pharm_imgs[change['new']]
        pharm_slider.observe(on_slider_change, 'value')
            
        # init checkbox table for picking pharmacophore models
        df_panel = ipywidgets.HTML(pharm_models.style.set_table_attributes('class="table"').render())
        items_layout = ipywidgets.Layout(flex='1 1 auto', width='auto')
        box_layout = ipywidgets.Layout(overflow_x='hidden', 
                                    display='flex',
                                    flex_flow='column',
                                    width='125px',
                                    align_items='stretch')
        checkbox_panel = ipywidgets.VBox([ipywidgets.Button(description='Select', layout=items_layout), *chk_objs], layout=box_layout)

        # show interactive params table
        display(ipywidgets.HBox([
            ipywidgets.VBox([pharm_slider, pharm_viewer]), 
            ipywidgets.HBox([df_panel, checkbox_panel])
        ]))