import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"
import matplotlib.patches as patches


cmp_reaction_infos = [
    [['PTS', 'GLK'], 'GLC', 'G6P', '|v'],
    [['PGI'], 'G6P', 'F6P', '|v'],
    [['PGI_r'], 'F6P', 'G6P', '|^'],
    [['PFK'], 'F6P', 'FBP', '|v'],
    [['FBP'], 'FBP', 'F6P', '|^'],
    [['ALD'], ['FBP'], ['DHAP', 'G3P'], [['x_center', 'bottom']], [['x_center', 'top'], ['x_center', 'top']]],
    [['ALD_r'], ['DHAP', 'G3P'], ['FBP'], [['x_center', 'top'], ['x_center', 'top']], [['x_center', 'bottom']]],
    [['TPI'], 'G3P', 'DHAP', '-<'],
    [['TPI_r'], 'DHAP', 'G3P', '->'],
    [['GAPDH', 'PGK'], 'G3P', '3PG', '|v'],
    [['GAPDH_r', 'PGK_r'], '3PG', 'G3P', '|^'],
    [['PGM', 'ENO'], '3PG', 'PEP', '|v'],
    [['PGM_r', 'ENO_r'], 'PEP', '3PG', '|^'],
    [['PYK', 'PTS'], 'PEP', 'PYR', '|v'],
    [['PPS'], 'PYR', 'PEP', '|^'],
    [['PFL'], ['PYR'], ['ACCOA', 'FOR'], 
     [['right', 'y_center']], 
     [['left', 'top'], ['left', 'bottom']]],
    [['PDH'], 'PYR', 'ACCOA', '->'],
    [['G6PDH', 'PGL'], 'G6P', '6PGC', '->'],
    [['GND'], '6PGC', 'RU5P', '|v'],
    [['RPE'], 'RU5P', 'XU5P', '-<'],
    [['RPE_r'], 'XU5P', 'RU5P', '->'],
    [['RPI'], 'RU5P', 'R5P', '|v'],
    [['RPI_r'], 'R5P', 'RU5P', '|^'],
    [['TKT1'], ['XU5P', 'R5P'], ['G3P', 'S7P'], 
     [['left', 'top'], ['left', 'top']], 
     [['right', 'top'], ['x_center', 'top']]],
    [['TKT1_r'], ['G3P', 'S7P'], ['XU5P', 'R5P'], 
     [['right', 'top'], ['x_center', 'top']], 
     [['left', 'top'], ['left', 'top']]],
    [['TAL'], ['S7P', 'G3P'], ['E4P', 'F6P'], 
     [['left', 'y_center'], ['right', 'bottom']], 
     [['left', 'top'], ['right', 'top']]],
    [['TAL_r'], ['E4P', 'F6P'], ['S7P', 'G3P'], 
     [['left', 'top'], ['right', 'top']], 
     [['left', 'y_center'], ['right', 'bottom']]],
    [['TKT2'], ['XU5P', 'E4P'], ['F6P', 'G3P'], 
     [['left', 'bottom'], ['left', 'bottom']], 
     [['right', 'bottom'], ['right', 'y_center']]],
    [['TKT2_r'], ['F6P', 'G3P'], ['XU5P', 'E4P'], 
     [['right', 'bottom'], ['right', 'y_center']], 
     [['left', 'bottom'], ['left', 'bottom']]],
    [['CS'], ['ACCOA', 'OAA'], ['CIT'], 
     [['left', 'bottom'], ['right', 'y_center']], 
     [['left', 'y_center']]],
    [['ACN1', 'ACN2'], 'CIT', 'ICT', '\v'],
    [['ACN1_r', 'ACN2_r'], 'ICT', 'CIT', '\^'],
    [['ICDH'], 'ICT', 'AKG', '|v'],
    [['SCS', 'AKGDH'], 'AKG', 'SUCC', '/-v'],
    [['SDH'], 'SUCC', 'FUM', '-<'],
    [['FRD'], 'FUM', 'SUCC', '->'],
    [['FUM'], 'FUM', 'MAL', '|^'],
    [['FUM_r'], 'MAL', 'FUM', '|v'],
    [['MDH'], 'MAL', 'OAA', '/^'],
    [['MDH_r'], 'OAA', 'MAL', '/v'],
    [['ME1', 'ME2'], 'MAL', 'PYR', '|^'],
    [['ICL'], ['ICT'], ['GLX', 'SUCC'], 
     [['left', 'y_center']], 
     [['right', 'y_center'], ['right', 'top']]],
    [['MS'], ['ACCOA', 'GLX'], ['MAL'], 
     [['x_center', 'bottom'], ['left', 'top']], 
     [['right', 'y_center']]],
    [['PPC'], 'PEP', 'OAA', '\|^'],
    [['PCK'], 'OAA', 'PEP', '\|v'],
    [['PTA', 'ACK'], 'ACCOA', 'AC', '->'],
    [['LDH'], 'PYR', 'LAC', '/-^'],
    [['ALDH', 'ADH'], 'ACCOA', 'ETOH', '->'],
    # [['R5Pt'], 'R5P', 'Export', '->'],
    # [['OAAt'], 'OAA', 'Export', '-<'],
    [['GLCt'], 'Import', 'GLC', '|v'],
    [['ACt'], 'AC', 'Export', '->'],
    [['SUCCt'], 'SUCC', 'Export', '|v'],
    [['FORt'], 'FOR', 'Export', '->'],
    [['LACt'], 'LAC', 'Export', '->'],
    [['ETOHt'], 'ETOH', 'Export', '->'],
]


class EColiCMPVisualizer:
    def __init__(self, reaction_infos=cmp_reaction_infos, 
                 ax=None, ax_left=-0.5, ax_right=0.65, 
                 ax_bottom=-0.81, ax_top=1.0, ax_scale=9, 
                 fontsize=8, boxsize=0.07):
        self.reaction_infos = reaction_infos
        if type(ax) == type(None):
            fig, ax = plt.subplots(
                1, 1, figsize=(ax_scale * (ax_right - ax_left), 
                               ax_scale * (ax_top - ax_bottom)),
            )
        ax.set_xlim(ax_left, ax_right)
        ax.set_ylim(ax_bottom, ax_top)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.set_facecolor((1,1,1,0))
        self.ax = ax
        self.fontsize = fontsize
        self.boxsize = boxsize

    def visualize_metabolites(self, color='grey'):
        ax = self.ax
        fontsize = self.fontsize
        boxsize = self.boxsize
        
        block = VerticalBlock(['GLC', 'G6P', 'F6P', 'FBP', 'G3P', '3PG', 'PEP', 'PYR'])
        # coordinates of visualized metabolites
        metab_locs = block.visualize(-0.2, 0.8, -0.2, ax, fontsize, boxsize=boxsize, color=color)

        TCA_metab_names = ['OAA', 'CIT', 'ICT', 'AKG', 'SUCC', 'FUM', 'MAL']
        block = CircleBlock(TCA_metab_names)
        TCA_center_xy = [0.0, -0.5]
        tmp = block.visualize(*TCA_center_xy, 0.2, np.pi/6, np.pi/6, ax, fontsize, boxsize=boxsize, color=color)
        metab_locs = dict(metab_locs, **tmp)

        x = metab_locs['PYR']['left']
        y = metab_locs['PYR']['bottom']
        block = HorizontalBlock(['ACCOA', 'AC'])
        tmp = block.visualize(x, x + 0.25, y, ax, fontsize, boxsize=boxsize, color=color, visualize_head=False)
        metab_locs = dict(metab_locs, **tmp)

        block = CircleBlock(['XU5P', 'S7P', 'E4P'])
        tmp = block.visualize(0.15, 0.4, 0.15, 10 * np.pi / 6, 10 * np.pi / 6, ax, fontsize, boxsize=boxsize, color=color)
        metab_locs = dict(metab_locs, **tmp)

        x = metab_locs['G6P']['left']
        y = metab_locs['G6P']['bottom']
        block = VerticalBlock(['6PGC', 'RU5P', 'R5P'])
        tmp = block.visualize(x + 0.6, y, y - 0.4, ax, fontsize, boxsize=boxsize, color=color)
        metab_locs = dict(metab_locs, **tmp)

        metab_name = 'GLX'
        p = visualize_box_with_text(*TCA_center_xy, metab_name, ax, boxsize=boxsize, fontsize=fontsize, color=color)
        metab_locs[metab_name] = return_box_loc(p)

        metab_name = 'ETOH'
        x = metab_locs['AC']['left']
        y = metab_locs['AC']['bottom']
        p = visualize_box_with_text(x, y - 0.2, metab_name, ax, boxsize=boxsize, fontsize=fontsize, color=color)
        metab_locs[metab_name] = return_box_loc(p)

        metab_name = 'DHAP'
        x = metab_locs['G3P']['left']
        y = metab_locs['G3P']['bottom']
        p = visualize_box_with_text(x - 0.2, y, metab_name, ax, boxsize=boxsize, fontsize=fontsize, color=color)
        metab_locs[metab_name] = return_box_loc(p)

        metab_name = 'FOR'
        x = metab_locs['ACCOA']['left']
        y = metab_locs['PEP']['bottom']
        p = visualize_box_with_text(x, y, metab_name, ax, boxsize=boxsize, fontsize=fontsize, color=color)
        metab_locs[metab_name] = return_box_loc(p)
        
        metab_name = 'LAC'
        x = metab_locs['ACCOA']['left']
        y = metab_locs['3PG']['bottom']
        p = visualize_box_with_text(x, y, metab_name, ax, boxsize=boxsize, fontsize=fontsize, color=color)
        metab_locs[metab_name] = return_box_loc(p)
        
        self.metab_locs = metab_locs

    def _visualize_reaction(self, reaction_info, **kwargs):
        ax = self.ax
        reaction = self.reaction
        multi_to_multi_reaction = self.multi_to_multi_reaction
        if type(reaction_info[1]) != list and type(reaction_info[2]) != list:
            reaction.visualize(*reaction_info[1:], ax, **kwargs)
        else:
            multi_to_multi_reaction.visualize(*reaction_info[1:], ax, 
                                              **kwargs)

    def visualize_reactions(self, kwargs_dict):
        ax = self.ax
        reaction_infos = self.reaction_infos
        metab_locs = self.metab_locs
        self.reaction = OneToOneReaction(metab_locs)
        self.multi_to_multi_reaction = MultiToMultiReaction(metab_locs)
        def return_kwargs(reaction_name):
            kwargs_keys = []
            for reaction_name_i in reaction_name:
                n_kwargs = 0
                if reaction_name_i in kwargs_dict.keys():
                    kwargs_key = reaction_name_i
                    n_kwargs += 1
                else:
                    kwargs_key = 'default'
                kwargs_keys.append(kwargs_key)
            if n_kwargs > 1:
                print('multiple kwargs')
                print(reaction_info)
            # which kws will be used
            kwargs = kwargs_dict[kwargs_keys[0]]
            return kwargs
        for reaction_info in reaction_infos:
            reaction_name = reaction_info[0]
            kwargs = return_kwargs(reaction_name)
            self._visualize_reaction(reaction_info, **kwargs)
        x1 = metab_locs['6PGC']['right']
        y1 = metab_locs['6PGC']['y_center']
        xy_starts = [[x1, y1]]
        xy_corners = []
        x2 = x1 + 0.05
        xy_corners.append([x2, y1])
        y2 = metab_locs['G3P']['bottom']
        xy_corners.append([x2, y2])
        x3 = metab_locs['G3P']['right'] + 0.2
        xy_corners.append([x3, y2])
        x4 = metab_locs['G3P']['right']
        x5 = metab_locs['PYR']['right']
        y3 = metab_locs['PYR']['top']
        xy_ends = [[x4, y2], [x5, y3]]
        self.EDD_EDA_reaction_info = [['EDD', 'EDA'], xy_starts, xy_ends, xy_corners]
        kwargs = return_kwargs(['EDD', 'EDA'])
        visualize_squared_arrow(xy_starts, xy_ends, xy_corners, 
                                ax, **kwargs)

    def color_metabolite(self, metab_name, color):
        ax = self.ax
        fontsize = self.fontsize
        boxsize = self.boxsize
        metab_locs = self.metab_locs
        x = metab_locs[metab_name]['left']
        y = metab_locs[metab_name]['bottom']
        p = visualize_box_with_text(x, y, metab_name, ax, boxsize=boxsize, fontsize=fontsize, color=color)


def return_box_loc(p):
    x_keys = ['left', 'x_center', 'right']
    y_keys = ['bottom', 'y_center', 'top']
    count = 0
    box_connect = {}
    for pos in [0.0, 0.5, 1.0]:
        x, y = p.get_patch_transform().transform((pos, pos))
        box_connect[x_keys[count]] = x
        box_connect[y_keys[count]] = y
        count += 1
    return box_connect


def visualize_box_with_text(x, y, text, ax, n_default_letter=3, boxsize=0.05, fontsize=10, color='blue', fill=False):
    p = patches.Rectangle(
            (x, y), boxsize, boxsize, color=color,
            fill=fill, clip_on=False
            # transform=ax.transAxes, 
        )
    ax.add_patch(p)
    text_rate = len(text) / n_default_letter
    ax.text(x + 0.5 * boxsize, y + 0.5 * boxsize, text,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=fontsize / text_rate, color='black')
    return p


class NetworkBlock():
    def __init__(self, metab_names):
        # metab_names = ['○' + metab_name for metab_name in metab_names]
        self.metab_names = metab_names


class VerticalBlock(NetworkBlock):
    def __init__(self, metab_names):
        super().__init__(metab_names)
    
    def visualize(self, x, y_start, y_end, ax, fontsize, color, boxsize=0.05, visualize_head=True):
        interval = (y_end - y_start) / (len(self.metab_names) - 1)
        metab_locs = {}
        for i, metab_name in enumerate(self.metab_names):
            j = None
            if visualize_head:
                j = i
            else:
                j = i + 1
            y = y_start + j * interval
            p = visualize_box_with_text(x, y, metab_name, ax, boxsize=boxsize, fontsize=fontsize, color=color)
            metab_locs[metab_name] = return_box_loc(p)
        return metab_locs


class HorizontalBlock(NetworkBlock):
    def __init__(self, metab_names):
        super().__init__(metab_names)
    
    def visualize(self, x_start, x_end, y, ax, fontsize, color, boxsize=0.05, visualize_head=True):
        interval = (x_end - x_start) / (len(self.metab_names) - 1)
        metab_locs = {}
        for i, metab_name in enumerate(self.metab_names):
            j = None
            if visualize_head:
                j = i
            else:
                j = i + 1
            x = x_start + j * interval
            p = visualize_box_with_text(x, y, metab_name, ax, boxsize=boxsize, fontsize=fontsize, color=color)
            metab_locs[metab_name] = return_box_loc(p)
        return metab_locs


class CircleBlock(NetworkBlock):
    def __init__(self, metab_names):
        super().__init__(metab_names)
    
    def visualize(self, center_x, center_y, r, 
                  theta_start, theta_end, ax, fontsize, color, boxsize=0.05):
        delta_theta = theta_end - theta_start
        if delta_theta < 0:
            delta_theta += 2 * np.pi
        elif delta_theta == 0:
            delta_theta = 2 * np.pi
        interval = delta_theta / len(self.metab_names)
        metab_locs = {}
        for i, metab_name in enumerate(self.metab_names):
            # clock-like
            theta = np.pi / 2 + theta_start - i * interval
            x, y = r * np.cos(theta), r * np.sin(theta)
            x += center_x
            y += center_y
            p = visualize_box_with_text(x, y, metab_name, ax, boxsize=boxsize, fontsize=fontsize, color=color)
            metab_locs[metab_name] = return_box_loc(p)
        return metab_locs


def visualize_arrow(xy_start, xy_end, ax, rate=1.0, 
                    width=0.01, head_length=None, head_width=None,
                    vertical_shift=0.0, horizontal_shift=0.0, **kwargs):
    # default head_length
    if head_length == None:
        # head_length = width * 4.5
        head_length = width * 2.5
    if head_width == None:
        if head_length == 0.0:
            head_width = 0.0
        else:
            # default head_width
            head_width = width * 3.0
    x = xy_start[0] + horizontal_shift
    y = xy_start[1] + vertical_shift
    dx = (xy_end[0] - xy_start[0]) * rate
    dy = (xy_end[1] - xy_start[1]) * rate
    ax.arrow(x, y, dx, dy, 
             width=width, length_includes_head=True, 
             head_length=head_length, 
             head_width=head_width, **kwargs)


joint_locs = {
    '|v': ['x_center', 'bottom', 'x_center', 'top'],
    '|^': ['x_center', 'top', 'x_center', 'bottom'],
    '->': ['right', 'y_center', 'left', 'y_center'],
    '-<': ['left', 'y_center', 'right', 'y_center'],
    '\v': ['right', 'bottom', 'left', 'top'],
    '\^': ['left', 'top', 'right', 'bottom'],
    '/v': ['left', 'bottom', 'right', 'top'],
    '/^': ['right', 'top', 'left', 'bottom'],
    '/-^': ['right', 'y_center', 'left', 'bottom'],
    '/-v': ['left', 'bottom', 'right', 'y_center'],
    '\|^': ['right', 'bottom', 'x_center', 'top'],
    '\|v': ['x_center', 'top', 'right', 'bottom']
}


class OneToOneReaction():
    def __init__(self, metab_locs):
        self.metab_locs = metab_locs

    def visualize(self, start_metab_name, end_metab_name, arrow_type, ax, 
                  arrow_length=0.1, **kwargs):
        x_start_key, y_start_key, x_end_key, y_end_key = joint_locs[arrow_type]
        # in case import/export reaction
        if start_metab_name == 'Import':
            x_end = self.metab_locs[end_metab_name][x_end_key]
            y_end = self.metab_locs[end_metab_name][y_end_key]
            if arrow_type == '|v':
                x_start = x_end
                y_start = y_end + arrow_length
        elif end_metab_name == 'Export':
            x_start = self.metab_locs[start_metab_name][x_start_key]
            y_start = self.metab_locs[start_metab_name][y_start_key]
            if arrow_type == '->':
                x_end = x_start + arrow_length
                y_end = y_start
            elif arrow_type == '-<':
                x_end = x_start - arrow_length
                y_end = y_start
            elif arrow_type == '|v':
                x_end = x_start
                y_end = y_start - arrow_length
            elif arrow_type == '|^':
                x_end = x_start
                y_end = y_start + arrow_length
        else:
            x_start = self.metab_locs[start_metab_name][x_start_key]
            y_start = self.metab_locs[start_metab_name][y_start_key]
            x_end = self.metab_locs[end_metab_name][x_end_key]
            y_end = self.metab_locs[end_metab_name][y_end_key]
        visualize_arrow([x_start, y_start], [x_end, y_end], 
                        ax, **kwargs)


def visualize_multi_to_multi_arrow(xy_starts, xy_ends, ax, **kwargs):
    all_xy = xy_starts + xy_ends
    mean_x, mean_y = np.array(all_xy).mean(0)
    no_head_kwargs = kwargs.copy()
    no_head_kwargs['head_length'] = 0.0
    for x, y in xy_starts:
        visualize_arrow([x, y], [mean_x, mean_y], ax,
                        **no_head_kwargs)
    for x, y in xy_ends:
        visualize_arrow([mean_x, mean_y], [x, y], ax,
                        **kwargs)


class MultiToMultiReaction():
    def __init__(self, metab_locs):
        self.metab_locs = metab_locs
    def visualize(self, start_metab_names, end_metab_names, start_joint_locs, end_joint_locs, 
                  ax, **kwargs):
        xy_starts = []
        xy_ends = []
        for start_metab_name, start_joint_loc in zip(start_metab_names, start_joint_locs):
            x = self.metab_locs[start_metab_name][start_joint_loc[0]]
            y = self.metab_locs[start_metab_name][start_joint_loc[1]]
            xy_starts.append([x, y])
        for end_metab_name, end_joint_loc in zip(end_metab_names, end_joint_locs):
            x = self.metab_locs[end_metab_name][end_joint_loc[0]]
            y = self.metab_locs[end_metab_name][end_joint_loc[1]]
            xy_ends.append([x, y])
        visualize_multi_to_multi_arrow(xy_starts, xy_ends, ax, **kwargs)


def visualize_squared_arrow(xy_starts, xy_ends, xy_corners, ax, **kwargs):
    no_head_kwargs = kwargs.copy()
    no_head_kwargs['head_length'] = 0.0
    if len(xy_starts) == 1:
        visualize_arrow(xy_starts[0], xy_corners[0], ax, **no_head_kwargs)
    else:
        visualize_multi_to_multi_arrow(xy_starts, [xy_corners[0]], ax, 
                                       **no_head_kwargs)
    for i in range(len(xy_corners)-1):
        visualize_arrow(xy_corners[i], xy_corners[i+1], ax, **no_head_kwargs)
    if len(xy_ends) == 1:
        visualize_arrow(xy_corners[-1], xy_ends[0], ax, **kwargs)
    else:
        visualize_multi_to_multi_arrow([xy_corners[-1]], xy_ends, ax, **kwargs)