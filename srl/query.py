import argparse
import json
import os
from collections import defaultdict
from pprint import pprint, pformat

from .uds import UDSCorpus


SPLITS = ["dev", "test", "train"]

AGENT_QUERY = """
    SELECT DISTINCT ?edge
    WHERE { ?edge
        <volition> ?volition ;
        <instigation> ?instigation ;
        <existed_before> ?existedBefore
        FILTER (
            (?volition > 0 || ?instigation > 0)
            && ?existedBefore > 0
        ) .
     }
"""

PATIENT_QUERY = """
    SELECT DISTINCT ?edge
    WHERE { ?edge
        <change_of_state> ?changeOfState ;
        <change_of_state_continuous> ?changeOfStateContinuous ;
        <existed_before> ?existedBefore ;
        <change_of_location> ?changeOfLocation
        FILTER (
            ?changeOfState > 0
            || ?changeOfStateContinuous > 0
            || ?changeOfLocation <= 0
            || ?existedBefore <= 0
        ) .
    }
"""

THEME_QUERY = """
    SELECT DISTINCT ?edge
    WHERE { ?edge
        <existed_before> ?existedBefore ;
        <change_of_state> ?changeOfState ;
        <change_of_location> ?changeOfLocation ;
        <existed_after> ?existedAfter
        FILTER (
            ?changeOfState > 0
            || ?changeOfLocation > 0
            || ?existedAfter <= 0
            || ?existedBefore <= 0
        ) .
    }
"""

EXPERIENCER_QUERY = """
    SELECT DISTINCT ?edge
    WHERE { ?edge
        <volition> ?volition ;
        <instigation> ?instigation ;
        <awareness> ?awareness ;
        <sentient> ?sentient ;
        <existed_before> ?existedBefore
        FILTER (
            (?volition <= 0 || ?instigation <= 0)
            && (?awareness > 0 || ?sentient > 0)
            && ?existedBefore > 0
        ) .
    }
"""

DESTINATION_QUERY = """
    SELECT DISTINCT ?edge
    WHERE { ?edge
        <existed_before> ?existedBefore ;
        <change_of_location> ?changeOfLocation ;
        <awareness> ?awareness ;
        <sentient> ?sentient ;
        <instigation> ?instigation ;
        <volition> ?volition
        FILTER (
            ?existedBefore > 0
            && ?changeOfLocation <= 0
            && (?instigation <= 0 || ?volition <= 0)
            && (?awareness <= 0 || ?sentient <= 0)
        ) .
    }
"""

PROTOROLE_TO_QUERY_MAP = {
    "agent": AGENT_QUERY,
    "patient": PATIENT_QUERY,
    "theme": THEME_QUERY,
    "experiencer": EXPERIENCER_QUERY,
    "destination": DESTINATION_QUERY
}


def get_graph_to_node_ids(uds_sparql_resp):
    return {k: list(v.keys()) for k, v in uds_sparql_resp.items()}


class UDSQuery:
    def __init__(self, protorole, split, limit=None, version="1.0"):
        uds = UDSCorpus(split=split, version=version)
        query = PROTOROLE_TO_QUERY_MAP[protorole]

        self._raw_results = uds.query(
            query,
            query_type="edge",
            cache_rdf=False,
            limit=limit
        )

        graph_to_node_ids = get_graph_to_node_ids(self._raw_results)

        self._results = []
        self._stats = defaultdict(int)

        def head_idx(head):
            idx, string = head
            return idx - 1, string

        for gid, node_ids in graph_to_node_ids.items():
            graph = uds[gid]
            syntax_edges = graph.syntax_edges()

            label = 1 if node_ids else 0

            def deprel(pred_id, arg_id):
                # There must be a better way to do this...
                syntax_pred_id = pred_id.replace("semantics-pred", "syntax")
                syntax_arg_id = arg_id.replace("semantics-arg", "syntax")
                key = (syntax_pred_id, syntax_arg_id)

                if key in syntax_edges:
                    return syntax_edges[key]["deprel"]

                # Sometimes semantic edges don't have corresponding syntactic
                # edges, as in the case of 'ewt-dev-2-semantics-pred-9' and
                # 'ewt-dev-2-semantics-arg-2' in 'ewt-dev-2'.
                return "NULL"

            # We need to retrieve the pred/arg pairs in results
            # without matches, but we only want the pairs that were
            # derived with predpatt.
            node_ids = node_ids if node_ids else [
                ids for ids, attrs in graph.argument_edges().items()
                if attrs["frompredpatt"]
            ]

            for pred_id, arg_id in node_ids:
                self._results.append({
                    "tokens": str(graph.sentence),
                    "deprel": deprel(pred_id, arg_id),
                    "pred_head": head_idx(graph.head(pred_id)),
                    "arg_head": head_idx(graph.head(arg_id)),
                    "label": label
                })

            self._stats[label] += len(node_ids)

    @property
    def results(self):
        return self._results

    @property
    def raw_results(self):
        return self._raw_results

    @property
    def stats(self):
        return self._stats

    @property
    def json_dump(self):
        return json.dumps(self.results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the UDS.")
    parser.add_argument("protorole", type=str,
                        choices=["agent", "patient", "theme",
                                 "experiencer", "destination"],
                        help="Protorole to query for.")
    parser.add_argument("--split", type=str,
                        choices=SPLITS,
                        help="Data split to query on.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stats",
                        action="store_true",
                        help="Show some stats about the query.")
    parser.add_argument("--pretty",
                        action="store_true",
                        help="Pretty print the query results.")
    parser.add_argument("--raw",
                        action="store_true",
                        help="Return the raw SPARQL responses.")
    parser.add_argument("--uds_version", type=str,
                        default="1.0",
                        choices=["1.0", "2.0"])

    args = parser.parse_args()

    if (args.split):
        query = UDSQuery(
            args.protorole, args.split,
            limit=args.limit, version=args.uds_version)

        if args.raw:
            pprint(query.raw_results)
        elif args.pretty:
            pprint(query.results)

        if args.stats:
            pprint(query.stats)
    else:
        stats = {}

        DATA_DIR = os.path.join(os.getcwd(), "data", args.protorole)
        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)

        for split in SPLITS:
            split_path = os.path.join(DATA_DIR, split)

            query = UDSQuery(
                args.protorole, split,
                version=args.uds_version)

            with open(split_path, "w+") as f:
                f.write(query.json_dump)

            stats[split] = query.stats

        stats_file = os.path.join(DATA_DIR, "stats")
        with open(stats_file, "w+") as f:
            f.write(pformat(stats))
