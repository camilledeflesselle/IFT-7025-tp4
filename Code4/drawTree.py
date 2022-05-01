# Fonction auxiliaire qui permet d'afficher un graphe avec la disposition d'un arbre
import networkx as nx 
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    '''
    Si le graphique est un arbre, cette fonction renverra les positions pour le tracer dans une
    disposition hiérarchique.
    Paramètres:
        G: le graphe (doit être un arbre)
        root: le nœud racine de la branche actuelle
        largeur: espace horizontal alloué à cette branche - évite le chevauchement avec d'autres branches
        vert_gap: écart entre les niveaux de la hiérarchie
        vert_loc: emplacement vertical de la racine
        xcenter: emplacement horizontal de la racine
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1, vert_gap = 10, vert_loc = 0, xcenter = 0, pos = None, parent = None):
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) *2
            nextx = xcenter - width- dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos  

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

