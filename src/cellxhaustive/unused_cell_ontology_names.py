"""
AT. Add general description here.
"""


# Import modules
import json
import re
# import numpy as np  # AT. Delete if np.sort is not needed



# AT. Add this in separate file to facilitate update?
# AT. json format --> Upload to dict with json.loads()? .tsv format?
# AT. Possible to automate this dict creation? For download as well?
# AT. Is this structure always the same? (lists as values)
# cell_markers_dict = {'http://purl.obolibrary.org/obo/CL_0000084': ['CD3+'],
#                      'http://purl.obolibrary.org/obo/CL_0000236': ['CD19+', 'CD20+', 'CD3-'],
#                      'http://purl.obolibrary.org/obo/CL_0000451': ['HLA-DR+', 'CD3-', 'CD19-', 'CD20-', 'CD14-', 'CD16-', 'CD56-'],
#                      'http://purl.obolibrary.org/obo/CL_0000545': ['CD3+', 'CD4+', 'CXCR3+', 'CCR6-', 'CD8-'],
#                      'http://purl.obolibrary.org/obo/CL_0000546': ['CD3+', 'CD4+', 'CD8-', 'CXCR3-'],
#                      'http://purl.obolibrary.org/obo/CL_0000576': ['CD14+', 'CD3-', 'CD19-', 'CD20-'],
#                      'http://purl.obolibrary.org/obo/CL_0000623': ['CD56+', 'CD3-'],
#                      'http://purl.obolibrary.org/obo/CL_0000624': ['CD3+', 'CD4+'],
#                      'http://purl.obolibrary.org/obo/CL_0000625': ['CD3+', 'CD8+'],
#                      'http://purl.obolibrary.org/obo/CL_0000782': ['HLA-DR+', 'CD11c+', 'CD3-', 'CD19-', 'CD20-', 'CD14-', 'CD16-', 'CD56-'],
#                      'http://purl.obolibrary.org/obo/CL_0000784': ['HLA-DR+', 'CD123+', 'CD3-', 'CD19-', 'CD20-', 'CD14-', 'CD16-', 'CD56-', 'CD11c-'],
#                      'http://purl.obolibrary.org/obo/CL_0000787': ['CD19+', 'CD20+', 'CD27+', 'CD3-'],
#                      'http://purl.obolibrary.org/obo/CL_0000788': ['CD19+', 'CD20+', 'IgD+', 'CD3-', 'CD27-'],
#                      'http://purl.obolibrary.org/obo/CL_0000792': ['CD3+', 'CD4+', 'CD127+-', 'CD25+'],
#                      'http://purl.obolibrary.org/obo/CL_0000814': ['CD3+', 'CD56+', 'CD14-', 'CD33-'],
#                      'http://purl.obolibrary.org/obo/CL_0000818': ['CD19+', 'CD20+', 'CD24++', 'CD38++', 'CD3-'],
#                      'http://purl.obolibrary.org/obo/CL_0000895': ['CD3+', 'CD4+', 'CCR7+', 'CD45RA+', 'CD8-'],
#                      'http://purl.obolibrary.org/obo/CL_0000899': ['CD3+', 'CD4+', 'CCR6+', 'CXCR3-', 'CD8-'],
#                      'http://purl.obolibrary.org/obo/CL_0000900': ['CD3+', 'CD8+', 'CCR7+', 'CD45RA+', 'CD4-'],
#                      'http://purl.obolibrary.org/obo/CL_0000904': ['CD3+', 'CD4+', 'CCR7+', 'CD8-', 'CD45RA-'],
#                      'http://purl.obolibrary.org/obo/CL_0000905': ['CD3+', 'CD4+', 'CD8-', 'CCR7-', 'CD45RA-'],
#                      'http://purl.obolibrary.org/obo/CL_0000907': ['CD3+', 'CD8+', 'CCR7+', 'CD4-', 'CD45RA-'],
#                      'http://purl.obolibrary.org/obo/CL_0000913': ['CD3+', 'CD8+', 'CD4-', 'CCR7-', 'CD45RA-'],
#                      'http://purl.obolibrary.org/obo/CL_0000917': ['CD3+', 'CD8+', 'CXCR3+'],
#                      'http://purl.obolibrary.org/obo/CL_0000938': ['CD56++', 'CD3-', 'CD19-', 'CD20-', 'CD14-', 'HLA-DR-', 'CD16-'],
#                      'http://purl.obolibrary.org/obo/CL_0000939': ['CD16+', 'CD56+', 'CD3-', 'CD19-', 'CD20-', 'CD14-', 'HLA-DR-'],
#                      'http://purl.obolibrary.org/obo/CL_0000970': ['CD19+', 'CD20+', 'CD27+', 'IgD+', 'CD3-'],
#                      'http://purl.obolibrary.org/obo/CL_0000980': ['CD19+', 'CD27++', 'CD38++', 'CD3-', 'CD20-'],
#                      'http://purl.obolibrary.org/obo/CL_0001043': ['CD3+', 'CD4+', 'CD38+', 'HLA-DR+', 'CD8-'],
#                      'http://purl.obolibrary.org/obo/CL_0001044': ['CD3+', 'CD4+', 'CD45RA+', 'CD8-', 'CCR7-'],
#                      'http://purl.obolibrary.org/obo/CL_0001045': ['CD3+', 'CD4+', 'CD127+-', 'CD25+', 'CCR4+', 'CD45RO-'],
#                      'http://purl.obolibrary.org/obo/CL_0001046': ['CD3+', 'CD4+', 'CD127+-', 'CD25+', 'CCR4+', 'CD45RO+'],
#                      'http://purl.obolibrary.org/obo/CL_0001048': ['CD3+', 'CD4+', 'CD127+-', 'CD25+', 'CCR4+', 'HLA-DR+'],
#                      'http://purl.obolibrary.org/obo/CL_0001049': ['CD3+', 'CD8+', 'CD38+', 'HLA-DR+', 'CD4-'],
#                      'http://purl.obolibrary.org/obo/CL_0001050': ['CD3+', 'CD8+', 'CD45RA+', 'CD4-', 'CCR7-'],
#                      'http://purl.obolibrary.org/obo/CL_0001052': ['CD3+', 'CD4-', 'CD8+', 'CXCR3-', 'CCR6-'],
#                      'http://purl.obolibrary.org/obo/CL_0001053': ['CD19+', 'CD20+', 'CD27+', 'CD3-', 'IgD-'],
#                      'http://purl.obolibrary.org/obo/CL_0002057': ['CD14+', 'CD3-', 'CD19-', 'CD20-', 'CD16-'],
#                      'http://purl.obolibrary.org/obo/CL_0002128': ['CD3+', 'CD8+', 'CCR6+'],
#                      'http://purl.obolibrary.org/obo/CL_0002397': ['CD14+', 'CD16+', 'CD3-', 'CD19-', 'CD20-']
#                      'http://purl.obolibrary.org/obo/CL_2000001': ['']  # AT. Add PBMC markers
#                      }

# AT. Likewise: separate file in json/tsv format?
# AT. Possible to automate this dict creation? For download as well?
# AT. Is this structure always the same? (2 elements tuple as values)
# cell_labels_dict = {'http://purl.obolibrary.org/obo/CL_0000084': ('T: T cell', 'T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000236': ('B: B cell', 'B cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000451': ('DC: dendritic cell', 'dendritic cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000545': ('T: Th1 CD4+ T cell', 'T-helper 1 cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000546': ('T: non-Th1/Th17 CD4+ T cell', 'T-helper 2 cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000576': ('M: monocyte', 'monocyte'),
#                  'http://purl.obolibrary.org/obo/CL_0000623': ('NK: NK cell', 'natural killer cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000624': ('T: CD4+ T cell', 'CD4-positive, alpha-beta T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000625': ('T: CD8+ T cell', 'CD8-positive, alpha-beta T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000782': ('DC: myeloid dendritic cell', 'myeloid dendritic cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000784': ('DC: plasmacytoid dendritic cell', 'plasmacytoid dendritic cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000787': ('B: memory B cell', 'memory B cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000788': ('B: naive B cell', 'naive B cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000792': ('T: Treg',
#                                                                'CD4-positive, CD25+, alpha-beta regulatory T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000814': ('T: NK T cell', 'mature NK T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000818': ('B: transitional B cell', 'transitional stage B cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000895': ('T: naive CD4+ T cell', 'naive thymus-derived CD4-positive, alpha-beta T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000899': ('T: Th17 CD4+ T cell', 'T-helper 17 cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000900': ('T: naive CD8+ T cell', 'naive thymus-derived CD8-positive, alpha-beta T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000904': ('T: central memory CD4+ T cell', 'central memory CD4-positive, alpha-beta T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000905': ('T: effector memory CD4+ T cell', 'effector memory CD4-positive, alpha-beta T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000907': ('T: central memory CD8+ T cell', 'central memory CD8-positive, alpha-beta T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000913': ('T: effector memory CD8+ T cell', 'effector memory CD8-positive, alpha-beta T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000917': ('T: Tc1 CD8+ T cell', 'Tc1 cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000938': ('NK: CD16- CD56bright NK cell', 'CD16-negative, CD56-bright natural killer cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000939': ('NK: CD16+ CD56+ NK cell', 'CD16-positive, CD56-dim natural killer cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000970': ('B: IgD+ memory B cell', 'unswitched memory B cell'),
#                  'http://purl.obolibrary.org/obo/CL_0000980': ('B: plasmablast', 'plasmablast'),
#                  'http://purl.obolibrary.org/obo/CL_0001043': ('T: activated CD4+ T cell', 'activated CD4-positive, alpha-beta T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0001044': ('T: effector CD4+ T cell', 'effector CD4-positive, alpha-beta T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0001045': ('T: naive CCR4+ Treg', 'naive CCR4-positive regulatory T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0001046': ('T: memory CCR4+ Treg', 'memory CCR4-positive regulatory T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0001048': ('T: activated CCR4+ Treg',
#                                                                'activated CD4-positive, CD25-positive, CCR4-positive, alpha-beta regulator T cell, human'),
#                  'http://purl.obolibrary.org/obo/CL_0001049': ('T: activated CD8+ T cell', 'activated CD8-positive, alpha-beta T cell, human'),
#                  'http://purl.obolibrary.org/obo/CL_0001050': ('T: effector CD8+ T cell', 'effector CD8-positive, alpha-beta T cell'),
#                  'http://purl.obolibrary.org/obo/CL_0001052': ('T: non-Tc1/Tc17 CD8+ T cell', 'Tc2 cell'),
#                  'http://purl.obolibrary.org/obo/CL_0001053': ('B: IgD- memory B cell', 'IgD-negative memory B cell'),
#                  'http://purl.obolibrary.org/obo/CL_0002057': ('M: CD16- monocyte', 'CD14-positive, CD16-negative classical monocyte'),
#                  'http://purl.obolibrary.org/obo/CL_0002128': ('T: Tc17 CD8+ T cell', 'Tc17 cell'),
#                  'http://purl.obolibrary.org/obo/CL_0002397': ('M: CD16+ monocyte', 'CD14-positive, CD16-positive monocyte'),
#                  'http://purl.obolibrary.org/obo/CL_2000001': ('PBMC', 'peripheral blood mononuclear cell')
#                  }

# Load files into dictionaries
with open('../data/config/cell_markers.json') as in_cell_markers:
    cell_markers_dict = json.load(in_cell_markers)

with open('../data/config/cell_labels.json') as in_cell_labels:
    cell_labels_dict = json.load(in_cell_labels)

major_cell_types = {}

for url, cell_markers in cell_labels_dict.items():
    # print(url)
    cell_type = cell_labels_dict[url][0].split(': ')[1]
    # major_cell_types[cell_type] = cell_markers_dict[url]
    try:
        major_cell_types[cell_type] = cell_markers_dict[url]
    except:
        print('OK')

with open('major_cell_types.json', 'w') as outfile:
    json.dump(major_cell_types, outfile, indent=4)



for url, cell_markers in cell_markers_dict.items():
    cell_type = cell_labels_dict[url][0].split(': ')[1]
    # AT. Why element 0?
    # cell_type = re.sub('bright', '+', cell_type)  # AT. Remove bright? Ask Bernat
    positive = []
    negative = []
    for marker in cell_markers:
        marker = re.sub(r'\+\+$', '+', re.sub(r'\+\-$', '+', marker))
        # AT. Is it really possible to have a protein name with '++-'?
        # If not, best to use the following line:
        # marker = re.sub(r'\+\+$|\+\-$', '+', marker)  # Replaces double-signs by '+'
        sign = marker[-1]
        protein = marker[:-1]
        if sign == '+':
            positive += [protein]
        else:
            negative += [protein]
        # Sort marker lists
        positive.sort()
        negative.sort()
        major_cell_types[cell_type] = {'positive': positive, 'negative': negative}
