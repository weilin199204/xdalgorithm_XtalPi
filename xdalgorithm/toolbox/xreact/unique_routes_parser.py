from PIL import Image, ImageDraw
from rdkit.Chem import Draw
from rdkit import Chem
import sys
import subprocess
import tempfile
import json

def crop_image(img, margin=5):
    """
    Crop an image by removing white space around it

    :param img: the image to crop
    :type img: PIL image
    :param margin: padding, defaults to 20
    :type margin: int, optional
    :return: the cropped image
    :rtype: PIL image
    """
    # First find the boundaries of the white area
    x0_lim = img.width
    y0_lim = img.height
    x1_lim = 0
    y1_lim = 0
    for x in range(0, img.width):
        for y in range(0, img.height):
            if img.getpixel((x, y)) != (255, 255, 255):
                if x < x0_lim:
                    x0_lim = x
                if x > x1_lim:
                    x1_lim = x
                if y < y0_lim:
                    y0_lim = y
                if y > y1_lim:
                    y1_lim = y
    x0_lim = max(x0_lim, 0)
    y0_lim = max(y0_lim, 0)
    x1_lim = min(x1_lim + 1, img.width)
    y1_lim = min(y1_lim + 1, img.height)
    # Then crop to this area
    cropped = img.crop((x0_lim, y0_lim, x1_lim, y1_lim))
    # Then create a new image with the desired padding
    out = Image.new(
        img.mode,
        (cropped.width + 2 * margin, cropped.height + 2 * margin),
        color="white",
    )
    out.paste(cropped, (margin + 1, margin + 1))
    return out

def draw_rounded_rectangle(img, color, arc_size=20):
    """
    Draw a rounded rectangle around an image

    :param img: the image to draw upon
    :type img: PIL image
    :param color: the color of the rectangle
    :type color: tuple or str
    :param arc_size: the size of the corner, defaults to 20
    :type arc_size: int, optional
    :return: the new image
    :rtype: PIL image
    """
    x0, y0, x1, y1 = img.getbbox()
    x1 -= 1
    y1 -= 1
    copy = img.copy()
    draw = ImageDraw.Draw(copy)
    arc_size_half = arc_size // 2
    draw.arc((x0, y0, arc_size, arc_size), start=180, end=270, fill=color)
    draw.arc((x1 - arc_size, y0, x1, arc_size), start=270, end=0, fill=color)
    draw.arc((x1 - arc_size, y1 - arc_size, x1, y1), start=0, end=90, fill=color)
    draw.arc((x0, y1 - arc_size, arc_size, y1), start=90, end=180, fill=color)
    draw.line((x0 + arc_size_half, y0, x1 - arc_size_half, y0), fill=color)
    draw.line((x1, arc_size_half, x1, y1 - arc_size_half), fill=color)
    draw.line((arc_size_half, y1, x1 - arc_size_half, y1), fill=color)
    draw.line((x0, arc_size_half, x0, y1 - arc_size_half), fill=color)
    return copy

def molecule_to_image(mol, frame_color,img_size=(300, 300)):
    """
    Create a pretty image of a molecule,
    with a colored frame around it

    :param mol: the molecule
    :type mol: Molecule
    :param frame_color: the color of the frame
    :type frame_color: tuple of int or str
    :return: the produced image
    :rtype: PIL image
    """
    mol = Chem.MolFromSmiles(mol)

    d2d = Draw.rdMolDraw2D.MolDraw2DCairo(*img_size)
    options = d2d.drawOptions()
    options.minFontSize = 25
    img = Draw.MolToImage(mol, options=options)

    return draw_rounded_rectangle(img, frame_color)


class Parser:
    def __init__(self, config):
        self.config = config  # dict
        self.linked_node = None
        self._graph_props = {"layout": "dot", "rankdir": "RL", "splines": "ortho"}
        self._mol_props = {"label": "", "color": "white", "shape": "none"}
        self._react_props = {
            "layout": "dot",
            "rankdir": "RL",
            "splines": "ortho"
        }

        self._link_node_template = {
            "fillcolor": "black",
            "shape": "circle",
            "style": "filled",
            "width": "0.1",
            "fixedsize": "true",
            "label": ""
        }
        self.imgs = []
        self._set_default_lines()
        self._set_default_node_lines()
        self._set_default_stride_lines()

    @classmethod
    def from_json(cls,file_path):
        with open(file_path) as json_file_handler:
            content = json_file_handler.read().replace("\n","").replace("\r","")
        assert len(content.strip()) > 0
        config = json.loads(content)
        return cls(config)

    @staticmethod
    def _get_props_strings(props):
        return ",\n\t\t".join(f'{key}="{value}"' for key, value in props.items())

    def add_molecule(self, molecule, frame_color):
        """
        Add a node that is a molecule to the graph.
        The image of the molecule will have a rectangular frame around it with a given color.
        """
        img = molecule_to_image(molecule, frame_color=frame_color)
        _, filename = tempfile.mkstemp(suffix=".png")
        img.save(filename)
        self._mol_props["image"] = filename
        self._node_lines.append(
            f"\t{id(molecule)} [{self._get_props_strings(self._mol_props)}\n\t];"
        )

    def add_edge(self, node1, node2):
        self._stride_lines.append(f'\t{id(node1)} -> {id(node2)} [arrowhead="none"];')

    def append_linked_node(self, molecule, reaction_type):
        linked_node = [1]
        self._node_lines.append(
            f"\t{id(linked_node)} [{self._get_props_strings(self._link_node_template)}\n\t];"
        )
        self._stride_lines.append(f'\t{id(linked_node)} -> {id(molecule)} [label="{reaction_type}"];\n\t')
        return linked_node

    def parse_route(self, route_lst, is_first):
        product = route_lst[1]
        reaction_type = route_lst[0]
        if is_first:
            self.add_molecule(product, frame_color='red')
        linked_node = self.append_linked_node(product, reaction_type)
        children_lst = []
        for element in route_lst[2:]:
            if isinstance(element, str):
                #print(element)
                self.add_molecule(element, frame_color='green')
                self.add_edge(element,linked_node)
            elif isinstance(element, list):
                #print(element[1])
                self.add_molecule(element[1], frame_color='yellow')
                #self.add_edge(linked_node, element[1])
                self.add_edge(element[1],linked_node)
                children_lst.append(element)
        #print("children list:", children_lst)
        for child in children_lst:
            self.parse_route(child, is_first=False)

    def _set_default_lines(self):
        self._lines = [
            'strict digraph {',
            f"\t graph [{self._get_props_strings(self._graph_props)}\n\t];"
        ]

    def _set_default_node_lines(self):
        self._node_lines = []

    def _set_default_stride_lines(self):
        self._stride_lines = []

    def _merge(self):
        self._stride_lines.reverse()
        self._lines = self._lines + self._node_lines + self._stride_lines
        self._lines.append("}")

    def export_dot_file(self, route, input_name):
        self._set_default_lines()
        self._set_default_node_lines()
        self._set_default_stride_lines()
        self.parse_route(route, is_first=True)
        self._merge()
        with open(input_name, "w") as fileobj:
            fileobj.write("\n".join(self._lines))

    def get_img(self, dot_name, png_name):
        ext = ".bat" if sys.platform.startswith("win") else ""
        subprocess.call([f"dot{ext}", dot_name, "-Tpng", "-o", png_name])
        route_img = Image.open(png_name)
        return route_img

    def run(self, file_prefix):
        if len(self.config) == 0:
            return
        for i, route in enumerate(self.config):
            dot_name = file_prefix + str(i) + ".dot"
            png_name = file_prefix + str(i) + ".png"
            self.export_dot_file(route, dot_name)
            img = self.get_img(dot_name, png_name)
            self.imgs.append(img)
        return len(self.config)