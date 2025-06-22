from graphviz import Digraph

def get_graph(root):
    nodes, edges = set(), set()
    def dfs(node):
        if node in nodes: return
        nodes.add(node)
        for child in node._prev:
            edges.add((child, node))
            dfs(child)
    dfs(root)
    return nodes, edges

def draw_graph(root):
    painter = Digraph(format='svg', graph_attr={'rankdir':'LR'})

    nodes, edges = get_graph(root) 
    for node in nodes:
        uid = str(id(node))

        painter.node(name=uid, label=f'{{{node.label} | {node.data} | {node.grad}}}', shape='record')
        if node._op:
            painter.node(name=uid+node._op, label=node._op)
            painter.edge(uid+node._op, uid)

    for child, parent in edges:
        painter.edge(str(id(child)), str(id(parent)) + parent._op)
    
    return painter