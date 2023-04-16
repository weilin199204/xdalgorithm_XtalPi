import ipywidgets as widgets
import time
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
import os

from ipywidgets import (
    HBox,
    VBox,
    Text,
    Output,
    Checkbox,
    Button,
    Dropdown,
    Layout
)
from rdkit import Chem
from IPython.display import display, HTML
from xdalgorithm.toolbox.xreact.utils import get_building_blocks_path
from xdalgorithm.toolbox.xreact.react_bot import ReactBot
from xdalgorithm.toolbox.xreact.unique_routes_parser import Parser
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)


class XreactGui:
    def __init__(self, setup=True):
        self.retro_reactor = None
        self._input_widgets = dict()
        self.imgs = None
        self._output_widgets = dict()
        self._buttons_widgets = dict()
        self._panels = []
        if setup:
            self.setup()

    def setup(self):
        self._create_input_widgets()
        self._create_routes_visual()
        self._display_panel()

    def __pywidget_to_dict(self, pyw):
        pyw_type = type(pyw)
        ret = {
            'pyw_type': pyw_type.__name__,
            'pyw_state': pyw.get_state(),
        }
        return ret

    def __dict_to_pywidget(self, dat):
        import sys
        pyw_type = dat['pyw_type']
        pyw_obj_state = dat['pyw_state']
        try:
            typeclass = getattr(sys.modules['ipywidgets'], pyw_type)
            pyw_obj = typeclass()
            pyw_obj.set_state(pyw_obj_state)
            return pyw_obj
        except Exception:
            logger.error('pywidget unpickle error!')

    def __deepcopy_exclude(self, exclude_attrs=None):
        import copy
        new_dict = {}
        for i in self.__dict__:
            if exclude_attrs:
                if i in exclude_attrs:
                    if isinstance(self.__dict__[i], dict):
                        new_dict[i] = {}
                    elif isinstance(self.__dict__[i], list):
                        new_dict[i] = []
                    else:
                        new_dict[i] = None
                else:
                    new_dict[i] = copy.deepcopy(self.__dict__[i])
        return new_dict

    def __getstate__(self):
        state = self.__dict__.copy()
        ret_state = self.__deepcopy_exclude(
            exclude_attrs=['_input_widgets', '_output_widgets', '_buttons_widgets', '_panels', 'delegate'])
        # ret_state['delegate'] = self.delegate
        # widgets list serialization
        pywidgets_attrs = ['_input_widgets', '_output_widgets', '_buttons_widgets']
        for pyw_attr_key in pywidgets_attrs:
            for i in state[pyw_attr_key]:
                pyw = state[pyw_attr_key][i]
                ret_state[pyw_attr_key][i] = self.__pywidget_to_dict(pyw)

        # panels list serialization
        for idx, widget in enumerate(state['_panels']):
            ret_state['_panels'].append(self.__pywidget_to_dict(widget))

        return ret_state

    def __setstate__(self, newstate):
        # widgets list deserialization
        pywidgets_attrs = ['_input_widgets', '_output_widgets', '_buttons_widgets']
        for pyw_attr_key in pywidgets_attrs:
            for i in newstate[pyw_attr_key]:
                pyw_dict = newstate[pyw_attr_key][i]
                newstate[pyw_attr_key][i] = self.__dict_to_pywidget(pyw_dict)

        # panels list deserialization
        for idx, widget in enumerate(newstate['_panels']):
            newstate['_panels'][idx] = self.__dict_to_pywidget(widget)

        self.__dict__.update(newstate)

    def _create_input_widgets(self):
        text_layout = Layout(display='flex', flex_flow='row', width='100%')
        empty_o = Output(
            layout={"width": "3%", "height": "100%"}
        )
        #
        smiles_label = widgets.HTML(value='<b>SMILES</b>', layout=Layout(width="15%"))
        self._input_widgets["smiles_text"] = Text(continuous_update=False,
                                                  layout=Layout(width="85%", border="2px solid silver"))
        # A smiles box to receive user's compound smiles
        smiles_hbox = HBox([smiles_label, self._input_widgets["smiles_text"]], layout=text_layout)

        # A output area to visualize the compound
        self._output_widgets["smiles_output"] = Output(
            layout={"border": "2px solid silver", "width": "100%", "height": "180px"})

        core_label = widgets.HTML(value='<b>Core</b>', layout=Layout(width="10%"))
        self._input_widgets["core_text"] = Text(continuous_update=False,
                                                layout=Layout(width="90%", border="2px solid silver"))
        # A core box to receive user's core smiles
        core_hbox = HBox([core_label, self._input_widgets["core_text"]], layout=text_layout)

        self._output_widgets["core_output"] = Output(
            layout={"border": "2px solid silver", "width": "100%", "height": "180px"})

        # self._output_widgets["smiles_output"] is designed to visualized the compound.
        input_smiles_vbox = VBox([smiles_hbox, self._output_widgets["smiles_output"]],
                                 layout=Layout(width='48%', border="1px solide silver"))

        # self._output_widgets["core_output"] is designed to visualized the core.
        input_core_vbox = VBox([core_hbox, self._output_widgets["core_output"]],
                               layout=Layout(width='48%', border="1px solide silver"))

        structure_input_panel = HBox([input_smiles_vbox, empty_o, input_core_vbox],
                                     layout=Layout(width='100%'))
        self._panels.append(structure_input_panel)

        options_label = widgets.HTML(value='<b>Options</b>', layout=Layout(width="7%"))
        self._input_widgets["csp"] = Checkbox(value=True, description="reaction_specificity", indent=False,
                                              layout=Layout(width="20%"))
        self._input_widgets["csrc"] = Checkbox(value=True, description="core_has_single_reactive_center", indent=False,
                                               layout=Layout(width="40%"))

        self._buttons_widgets["run_retro_btn"] = Button(description="Retro_Analysis", layout=Layout(width="25%"))
        # listen to the function `self._on_exec_btn_clicked`
        self._buttons_widgets["run_retro_btn"].on_click(self._on_exec_btn_clicked)
        optional_panel = HBox([options_label, self._input_widgets["csp"], self._input_widgets["csrc"],
                               self._buttons_widgets["run_retro_btn"]], layout=Layout(width='100%'))
        self._panels.append(optional_panel)

        # add signal
        self._input_widgets["smiles_text"].observe(self._show_mol, names="value")
        self._input_widgets["core_text"].observe(self._show_core, names="value")

    def _create_routes_visual(self):
        options_label = widgets.HTML(value='<b>Routes</b>', layout=Layout(width="5%"))
        self._input_widgets["route"] = Dropdown(options=[])
        self._output_widgets["img_display"] = widgets.Output(
            layout={"border": "2px solid silver", "width": "99%", "height": "auto"})
        routes_imgs_panel = VBox(
            [HBox([options_label, self._input_widgets["route"]]), self._output_widgets["img_display"]])
        self._panels.append(routes_imgs_panel)

        # add signal
        self._input_widgets["route"].observe(self._on_change_route_option)

    def _display_panel(self):
        display(VBox(self._panels))

    def _show_mol(self, change):
        self._output_widgets["smiles_output"].clear_output()
        with self._output_widgets["smiles_output"]:
            mol = Chem.MolFromSmiles(change["new"])
            display(mol)

    def _show_core(self, change):
        self._output_widgets["core_output"].clear_output()
        with self._output_widgets["core_output"]:
            mol = Chem.MolFromSmiles(change["new"])
            display(mol)

    def _on_exec_btn_clicked(self, _):
        self._output_widgets["img_display"].clear_output()
        self._toggle_button(False)
        self._start_retro()
        self.imgs = self._query_imgs()
        self._input_widgets["route"].options = [i for i, j in enumerate(self.imgs)]

        if len(self.imgs) == 0:
            self._input_widgets["route"].options = ["No route is found"]
            self._show_no_found()
        else:
            self._input_widgets["route"].options = [i for i, j in enumerate(self.imgs)]

        self._toggle_button(True)

    # to get unique_routes
    def _start_retro(self):
        start_smiles = self._input_widgets["smiles_text"].value
        core_smiles = self._input_widgets["core_text"].value

        core_specific = self._input_widgets["csp"].value
        core_single_reactive_center = self._input_widgets["csrc"].value

        self.retro_reactor = ReactBot(start_smiles=start_smiles, core_smiles=core_smiles)
        self.retro_reactor.analysis(core_specific=core_specific,
                                    core_single_reactive_center=core_single_reactive_center)

    def _query_imgs(self):
        time_stamp = round(time.time() * 1000)
        file_prefix = "/tmp/parser_dot_file_{0}_".format(time_stamp)
        self.parser = Parser(self.retro_reactor.unique_routes)
        _ = self.parser.run(file_prefix)
        return self.parser.imgs

    def _toggle_button(self, on):
        for button in self._buttons_widgets.values():
            button.disabled = not on

    def _on_change_route_option(self, change):
        if change["name"] != "index":
            return
        self._show_route(self._input_widgets["route"].index)

    def _show_no_found(self):
        self._output_widgets["img_display"].clear_output()
        with self._output_widgets["img_display"]:
            display(HTML("<H2>No route is found."))

    def get_bb_name(self, route_index):
        return self.retro_reactor._runRoute(self.retro_reactor.unique_routes[route_index])

    def _show_route(self, route_index):
        if route_index is None:
            return

        self._output_widgets["img_display"].clear_output()
        with self._output_widgets["img_display"]:
            display(HTML("<H2>Routes"))
            display(self.imgs[route_index])
            display(HTML('<H1>BB Reactants:</H1><H2>' + '</H2><H2>'.join(self.get_bb_name(route_index))))

    def get_remote_bb_path(self, route_index, use_bbs_from_cas=False):
        if use_bbs_from_cas:
            bb_path = '/data/aidd-server/Building_blocks_from_chemical_books'
        else:
            bb_path = get_building_blocks_path()
        return [os.path.join(bb_path, p) for p in self.get_bb_name(route_index)]

    def get_reactants_def(self, route_index):
        return [p for p in self.get_bb_name(route_index)]

    def get_unique_route_list(self):
        return self.retro_reactor.unique_routes

    def get_unique_route(self, index):
        if index < len(self.retro_reactor.unique_routes):
            return self.retro_reactor.unique_routes[index]
        else:
            raise IndexError('index is out of range!')
