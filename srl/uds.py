from itertools import islice
from decomp import UDSCorpus as _UDSCorpus


class UDSCorpus(_UDSCorpus):
    def query(self, query,
              query_type=None,
              cache_query=True,
              cache_rdf=True,
              limit=None):
        """Query with an optional limit."""

        graphs = islice(self.items(), limit) if limit else self.items()

        return {
            gid: graph.query(
                query, query_type,
                cache_query, cache_rdf)
            for gid, graph in graphs}
