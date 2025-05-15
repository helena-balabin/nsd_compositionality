"""Preprocess VG data for Graph Image Model."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import amrlib
import hydra
import networkx as nx
import nltk
import pandas as pd
import penman
import spacy
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from nltk.corpus import wordnet as wn
from omegaconf import DictConfig
from spacy import Language
from tqdm import tqdm

logging.getLogger("penman").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
load_dotenv()


def preprocess_split(vg_metadata, nsd_coco_ids, vg_metadata_dir, cfg, split_name: str) -> Dict[str, Any]:
    """
    Preprocess a split of the VG metadata.

    :param vg_metadata: The VG metadata to preprocess.
    :param nsd_coco_ids: The set of NSD COCO IDs to filter against.
    :param vg_metadata_dir: The directory containing VG metadata files.
    :param cfg: The Hydra configuration object.
    :param split_name: The name of the split (e.g., "train" or "test").
    :return: The preprocessed VG metadata.
    """
    if split_name == "train":
        vg_metadata = vg_metadata.filter(
            lambda x: x[cfg.data.coco_id_col] not in nsd_coco_ids,
            num_proc=4,
        )
    elif split_name == "test":
        vg_metadata = vg_metadata.filter(
            lambda x: x[cfg.data.coco_id_col] in nsd_coco_ids,
            num_proc=4,
        )
    logger.info(f"Filtered VG metadata for {split_name} split, total entries: {len(vg_metadata)}")

    # Add image graph properties if specified in the config
    if cfg.data.include_image_graphs:
        vg_objects_file = vg_metadata_dir / "objects.json"
        vg_relationships_file = vg_metadata_dir / "relationships.json"
        graphs, filtered_graphs = derive_image_graphs(
            vg_objects_file=vg_objects_file,
            vg_relationships_file=vg_relationships_file,
            image_ids=vg_metadata[cfg.data.vg_image_id_col],
        )
        graphs = [graphs[img_id] for img_id in vg_metadata[cfg.data.vg_image_id_col]]  # type: ignore
        filtered_graphs = [filtered_graphs[img_id] for img_id in vg_metadata[cfg.data.vg_image_id_col]]  # type: ignore
        vg_metadata = vg_metadata.add_column(name="image_graphs", column=graphs)
        vg_metadata = vg_metadata.add_column(name="filtered_image_graphs", column=filtered_graphs)

    # Add text graph properties if specified in the config
    if cfg.data.include_text_graphs:
        texts = vg_metadata["sentences_raw"]
        text_ids = vg_metadata["sentids"]
        amr_graphs, dependency_graphs = derive_text_graphs(
            texts=texts,
            text_ids=text_ids,
            spacy_model=cfg.model.spacy_model,
        )
        amr_graphs = [amr_graphs[tid] for tid in text_ids]  # type: ignore
        dependency_graphs = [dependency_graphs[tid] for tid in text_ids]  # type: ignore
        vg_metadata = vg_metadata.add_column(name="amr_graphs", column=amr_graphs)
        vg_metadata = vg_metadata.add_column(name="dependency_graphs", column=dependency_graphs)

    return vg_metadata


@hydra.main(config_path="../../../configs/data", config_name="data")
def preprocess_vg_for_graphormer(cfg: DictConfig) -> None:
    """
    Preprocess VG data for the Graph Image Model.

    Args:
        cfg (DictConfig): The configuration object loaded by Hydra.
    """
    large_data_path = Path(cfg.data.large_data_path)

    nsd_dir = large_data_path / cfg.data.nsd_directory
    vg_dir = large_data_path / cfg.data.vg_directory
    vg_metadata_dir = vg_dir / cfg.data.vg_metadata_dir

    # Load NSD metadata
    nsd = pd.read_csv(nsd_dir / "nsd_vg" / "nsd_vg_metadata.csv")
    nsd_coco_ids = set(nsd["cocoId"].tolist())

    # Load VG metadata
    vg_metadata = load_dataset(
        cfg.data.vg_metadata_hf_identifier,
        cache_dir=cfg.data.cache_dir,
        split="train",
    )

    # Preprocess train split
    train_metadata = preprocess_split(
        vg_metadata=vg_metadata,
        nsd_coco_ids=nsd_coco_ids,
        vg_metadata_dir=vg_metadata_dir,
        cfg=cfg,
        split_name="train",
    )

    # Preprocess test split
    test_metadata = preprocess_split(
        vg_metadata=vg_metadata,
        nsd_coco_ids=nsd_coco_ids,
        vg_metadata_dir=vg_metadata_dir,
        cfg=cfg,
        split_name="test",
    )

    # Push both splits to the Hugging Face Hub
    dataset_dict = DatasetDict({"train": train_metadata, "test": test_metadata})
    dataset_dict.push_to_hub(
        repo_id=cfg.data.processed_hf_identifier,
    )


def derive_image_graphs(
    vg_objects_file: str,
    vg_relationships_file: str,
    image_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Get the graph data of the VG + COCO overlap dataset for the given image ids.

    :param vg_objects_file: Path to the file where the Visual Genome objects json is stored.
    :type vg_objects_file: str
    :param vg_relationships_file: Path to the file where the Visual Genome relationship json is stored.
    :type vg_relationships_file: str
    :param image_ids: Optional list of image ids to characterize the graph complexity for, defaults to None
    :type image_ids: Optional[List[str]]
    :param return_graphs: Whether to return the graphs as well, defaults to False
    :type return_graphs: bool
    :return: Two dictionaries with the graph complexity measures and image id
    :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
    """
    # Load the object and relationship files from json
    vg_objects = load_dataset("json", data_files=str(vg_objects_file), split="train")
    vg_relationships = load_dataset("json", data_files=str(vg_relationships_file), split="train")
    # Filter by image ids if given
    if image_ids:
        vg_objects = vg_objects.filter(lambda x: x["image_id"] in image_ids, num_proc=4)
        vg_relationships = vg_relationships.filter(
            lambda x: x["image_id"] in image_ids,
            num_proc=4,
        )

    # Process each VG image/graph into a networkx graph
    graphs = {}
    filtered_graphs = {}
    for obj, rel in tqdm(
        zip(vg_objects, vg_relationships),
        desc="Processing rels/objs as networkx graphs",
        total=len(vg_objects),
    ):
        # Create the graph based on objects and relationships
        graph = nx.DiGraph()
        for o in obj["objects"]:
            graph.add_node(o["object_id"])
        for r in rel["relationships"]:
            graph.add_edge(
                r["object"]["object_id"],
                r["subject"]["object_id"],
                rel_id=r["relationship_id"],
            )

        # Append the graph to the dict
        graphs[obj["image_id"]] = graph
        # Filter for relationships that have at least one living being as subject/object
        filtered_rels = [
            r
            for r in rel["relationships"]
            if len(r["object"]["synsets"]) > 0
            and len(r["subject"]["synsets"]) > 0
            and (check_if_living_being(r["object"]["synsets"][0]) or check_if_living_being(r["subject"]["synsets"][0]))
        ]
        filtered_rel_ids = [r["relationship_id"] for r in filtered_rels]
        filtered_edges = [
            (u, v, data) for u, v, data in graph.edges(data=True) if data.get("rel_id") in filtered_rel_ids
        ]
        # Create a new graph with the filtered edges
        filtered_graph = nx.DiGraph(filtered_edges)
        filtered_graphs[obj["image_id"]] = filtered_graph

    # Calculate the graphormer attributes
    graphs_graphormer = {}
    filtered_graphs_graphormer = {}
    for (graph_id, graph), (filtered_graph_id, filtered_graph) in tqdm(
        zip(graphs.items(), filtered_graphs.items()),
        desc="Calculating graphormer attributes",
        total=len(graphs),
    ):
        graphs_graphormer[graph_id] = calculate_graphormer_attributes(graph)
        filtered_graphs_graphormer[filtered_graph_id] = calculate_graphormer_attributes(filtered_graph)

    return graphs_graphormer, filtered_graphs_graphormer


def check_if_living_being(
    synset: str,
) -> bool:
    """Check if a given synset is a living being by recursively checking its hypernyms.

    :param synset: The synset to check, e.g., "dog.n.01"
    :type synset: str
    :return: True if the synset describes a living being
    :rtype: bool
    """
    if len(synset) == 0:
        return False
    synset = wn.synset(synset)
    hypernyms = set()

    def recursive_hypernyms(
        syn: nltk.corpus.reader.wordnet.Synset,  # type: ignore
    ):
        """Recursively check the hypernyms of a given synset.

        :param syn: The synset to check
        :type syn: wn.Synset
        """
        for hypernym in syn.hypernyms():
            hypernyms.add(hypernym)
            recursive_hypernyms(hypernym)

    recursive_hypernyms(synset)
    return wn.synset("animal.n.01") in hypernyms or wn.synset("person.n.01") in hypernyms


def calculate_graphormer_attributes(graph: nx.Graph) -> Dict[str, Any]:
    """
    Calculate the edge_index and num_nodes for a given networkx graph.

    :param graph: The input graph
    :type graph: nx.Graph
    :return: A dictionary containing edge_index and num_nodes
    :rtype: Dict[str, Any]
    """
    # Create a mapping from original node IDs to sequential IDs starting from 0
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}

    # Relabel the nodes in the graph using the mapping
    graph = nx.relabel_nodes(graph, node_mapping)

    # Convert edge_index to a 2 x n_edges format
    edge_index = [[u, v] for u, v in list(graph.edges())]
    edge_index = list(map(list, zip(*edge_index)))  # Transpose to 2 x n_edges format
    # If there are no edges, create an empty edge_index
    if len(edge_index) == 0:
        edge_index = [[], []]

    # Get the number of nodes
    num_nodes = graph.number_of_nodes()

    return {"edge_index": edge_index, "num_nodes": num_nodes}


def tree_to_graph(tree, start_index: int = 0) -> Tuple[nx.DiGraph, int]:
    """
    Recursively convert an nltk.Tree into a networkx.DiGraph.
    Returns the graph and the next available node id.
    """
    g = nx.DiGraph()
    current_idx = start_index
    node_id = current_idx
    label = tree.label() if hasattr(tree, "label") else tree
    g.add_node(node_id, label=label)
    current_idx += 1
    for child in tree:
        if isinstance(child, nltk.Tree):
            child_graph, current_idx = tree_to_graph(child, current_idx)
            child_root = min(child_graph.nodes())
            g.add_edge(node_id, child_root)
            g = nx.compose(g, child_graph)
        else:
            child_id = current_idx
            g.add_node(child_id, label=child)
            g.add_edge(node_id, child_id)
            current_idx += 1
    return g, current_idx


def derive_text_graphs(
    texts: List[str],
    text_ids: List[int],
    spacy_model: str = "en_core_web_trf",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Get graphs derived from texts: AMR graphs (using amrlib), dependency parse trees and constituency parse trees.

    Expects a JSON file (loaded as a Hugging Face dataset) where each entry contains:
      - "text_id": a unique identifier for the text.
      - "text": the text to process.

    :param texts: List of texts to process.
    :type texts: List[str]
    :param text_ids: List of unique identifiers for the texts.
    :type text_ids: List[int]
    :param spacy_model: The spaCy model to use for parsing, defaults to "en_core_web_trf".
    :type spacy_model: str
    :return: Two dictionaries (amr_graphs, dependency_graphs).
    :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
    """

    @Language.component("force_single_sentence")
    def one_sentence_per_doc(
        doc: spacy.tokens.Doc,  # noqa
    ) -> spacy.tokens.Doc:  # noqa
        """Force the document to be one sentence.

        :param doc: The document to force to be one sentence
        :type doc: spacy.tokens.Doc
        :return: The document with one sentence
        :rtype: spacy.tokens.Doc
        """
        doc[0].sent_start = True
        for i in range(1, len(doc)):
            doc[i].sent_start = False
        return doc

    # Add dependency parse tree depth and AMR depth
    # Prefer GPU if available
    spacy.prefer_gpu()
    # Set the AMR extension
    amrlib.setup_spacy_extension()
    # Disable unnecessary components
    nlp = spacy.load(spacy_model, disable=["tok2vec", "attribute_ruler", "lemmatizer", "ner", "tagger"])
    nlp.add_pipe("force_single_sentence", before="parser")

    amr_graphs = {}
    dependency_graphs = {}

    for tid, text in tqdm(
        zip(text_ids, texts),
        desc="Processing texts for graph parsing",
        total=len(texts),
    ):
        # Process the text with spaCy
        doc = nlp(text)

        # Dependency Parse Graph
        dep_graph = nx.DiGraph()
        for token in doc:
            dep_graph.add_node(token.i)
        for token in doc:
            if token.head.i != token.i:
                dep_graph.add_edge(token.head.i, token.i, dep=token.dep_)
        dependency_graphs[tid] = calculate_graphormer_attributes(dep_graph)

        # AMR Graph
        amr_penman = doc._.to_amr()[0]
        # Convert the penman graph to a NetworkX DiGraph
        try:
            penman_graph = penman.decode(amr_penman)
        except Exception as err:  # noqa
            logger.warning(f"Failed to decode AMR graph for text ID {tid}")
            amr_graphs[tid] = {"edge_index": [[], []], "num_nodes": 0}
            continue
        # Convert to a nx graph, first initialize the nx graph
        nx_graph = nx.DiGraph()
        # Add edges
        for e in penman_graph.edges():
            nx_graph.add_edge(e.source, e.target)
        amr_graphs[tid] = calculate_graphormer_attributes(nx_graph)

    return amr_graphs, dependency_graphs


if __name__ == "__main__":
    preprocess_vg_for_graphormer()
