class Edge(object):
    def __init__(self, idx, i, j):
        self.idx = idx
        self.i = i
        self.j = j

    def is_tip(self, u):
        return (self.i == u) or (self.j == u)

    def get_other(self, u):
        if self.i == u:
            return self.j 
        if self.j == u:
            return self.i
        return None

    def is_local(self):
        return False

class GraphEdge(Edge):
    pass