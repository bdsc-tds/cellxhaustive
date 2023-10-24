import numpy as np

# A utility function to rename cell types
def find_set_differences(cell_groups_renaming, baseline_name = "baseline"):
    """Find key differences across dictionaries. Provided a dictionary where every key corresponds to a given cell line,
       and every value is the set of positive/negative markers for that same cell line,
       the function will provide a dictionary with the key diferences between cell types.

       Parameters
       ----------
       cell_groups_renaming : dict(set(str))
         Imagine that you have three cell lines (CD4 T, CD8 T and DCs) that are characterized by three markers: CD3, CD4, CD8.
         `cell_groups_renaming` is a dictionary such that `cell_groups_renaming["CD4 T"] = set(["CD3+", "CD4+", "CD8-"])`
         
       baseline_name : str
         name given to the baseline cell type picked.
       
       Returns
       -------
       Following the example above, the function will return the following dictionary:
       ```
       {'CD4 T': 'CD4+', 'CD8 T': 'CD8+', 'DCs': 'CD3-'}
       ```
    """
    mark, cnts = np.unique([y for x in cell_groups_renaming.keys() for y in list(cell_groups_renaming[x])], return_counts=True)
    common_mark = mark[np.flip(np.argsort(cnts))]

    # Identify what are the key markers that distinguish the different groups and define the baseline based on the shortest combination
    keep_marks = []
    common_first = set()
    for i in common_mark:
        x = i.replace("-", "").replace("+", "")
        if x not in common_first:
            common_first.add(x)
            keep_marks.append(i)
            
    for x in cell_groups_renaming.keys():
        cell_groups_renaming[x] = " ".join(np.sort(list(cell_groups_renaming[x] - set(keep_marks))))
        if cell_groups_renaming[x] == "" and x!=-1:
            cell_groups_renaming[x] = baseline_name
        elif x==-1:
            cell_groups_renaming[x] = "undefined"

    return cell_groups_renaming
