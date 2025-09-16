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
from datasets import Dataset, DatasetDict, Value, load_dataset
from dotenv import load_dotenv
from nltk.corpus import wordnet as wn
from omegaconf import DictConfig
from spacy import Language
from tqdm import tqdm

logging.getLogger("penman").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
load_dotenv()


def map_vg_to_clip_patch(x, y, width, height, patch_size=32, target_resolution=224):
    """
    Map VG object coordinates to CLIP patch indices.

    :param x: X coordinate of the object center in original image
    :param y: Y coordinate of the object center in original image
    :param width: Original image width
    :param height: Original image height
    :param patch_size: CLIP patch size (16 or 32)
    :param target_resolution: Target resolution for CLIP (default 224)
    :return: Patch index in the CLIP patch grid
    """
    # Step 1: Resize so that the shorter side is target_resolution (e.g. 224)
    if width < height:
        scale = target_resolution / width
        new_w = target_resolution
        new_h = int(round(height * scale))
    else:
        scale = target_resolution / height
        new_h = target_resolution
        new_w = int(round(width * scale))

    # Step 2: Resize original coordinates
    x_resized = x * scale
    y_resized = y * scale

    # Step 3: Center crop
    left = (new_w - target_resolution) / 2
    top = (new_h - target_resolution) / 2

    x_clip = x_resized - left
    y_clip = y_resized - top

    # Step 4: Clamp to valid pixel range
    x_clip = max(0, min(target_resolution - 1e-3, x_clip))
    y_clip = max(0, min(target_resolution - 1e-3, y_clip))

    # Step 5: Compute patch grid size (ViT expects square inputs, so grid is square)
    num_patches_per_side = target_resolution // patch_size

    # Step 6: Compute patch index (row-major order)
    patch_x = int(x_clip // patch_size)
    patch_y = int(y_clip // patch_size)

    patch_index = patch_y * num_patches_per_side + patch_x

    # Return patch index incremented by 1 to match CLIP's preceeding [CLS] token
    return patch_index + 1


def preprocess_split(
    vg_metadata,
    nsd_coco_ids,
    vg_metadata_dir,
    cfg,
    split_name: str,
    vg_coco_overlap: Optional[Dataset],
) -> Dict[str, Any]:
    """
    Preprocess a split of the VG metadata.

    :param vg_metadata: The VG metadata to preprocess.
    :param nsd_coco_ids: The set of NSD COCO IDs to filter against.
    :param vg_metadata_dir: The directory containing VG metadata files.
    :param cfg: The Hydra configuration object.
    :param split_name: The name of the split (e.g., "train" or "test").
    :param vg_coco_overlap: The VG-COCO overlap dataset containing text, if text is to be included.
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
        # Visual VerbNet is from the COCO actions dataset
        vg_visual_verbs_file = vg_metadata_dir / "visual_verbnet_beta2015.json"
        graphs, action_graphs, spatial_graphs = derive_image_graphs(
            vg_objects_file=vg_objects_file,
            vg_relationships_file=vg_relationships_file,
            vg_visual_verbs_file=vg_visual_verbs_file,
            vg_metadata_dir=vg_metadata_dir,
            cfg=cfg,
            image_ids=vg_metadata[cfg.data.vg_image_id_col],
        )
        graphs = [graphs[img_id] for img_id in vg_metadata[cfg.data.vg_image_id_col]]  # type: ignore
        action_graphs = [action_graphs[img_id] for img_id in vg_metadata[cfg.data.vg_image_id_col]]  # type: ignore
        spatial_graphs = [spatial_graphs[img_id] for img_id in vg_metadata[cfg.data.vg_image_id_col]]  # type: ignore
        vg_metadata = vg_metadata.add_column(name="image_graphs", column=graphs)
        vg_metadata = vg_metadata.add_column(name="action_image_graphs", column=action_graphs)
        vg_metadata = vg_metadata.add_column(name="spatial_image_graphs", column=spatial_graphs)

    # Include text if specified in the config
    if cfg.data.include_text:
        # Rename "cocoid" in "coco_id" if needed
        if vg_coco_overlap is not None:
            # For those entries in vg_metadata that have a matching entry in vg_coco_overlap, add the captions,
            # Add none otherwise
            vg_metadata_df = vg_metadata.to_pandas()
            if "cocoid" in vg_coco_overlap.column_names:
                vg_coco_overlap = vg_coco_overlap.rename_column("cocoid", "coco_id")
            vg_coco_overlap_df = vg_coco_overlap.to_pandas()
            merged_df = vg_metadata_df.merge(
                vg_coco_overlap_df[[cfg.data.coco_id_col, "sentences_raw", "sentids"]],
                on=cfg.data.coco_id_col,
                how="left",
            )
            vg_metadata = Dataset.from_pandas(merged_df)
            # Shuffle the dataset to mix entries with different captions but the same image
            vg_metadata = vg_metadata.shuffle(seed=cfg.seed)

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

    # If text should be added, load COCO captions
    if cfg.data.include_text:
        vg_coco_overlap = load_dataset(
            cfg.data.coco_text_hf_identifier,
            cache_dir=cfg.data.cache_dir,
            split="train",
        )
    else:
        vg_coco_overlap = None

    # Preprocess train split
    train_metadata = preprocess_split(
        vg_metadata=vg_metadata,
        nsd_coco_ids=nsd_coco_ids,
        vg_metadata_dir=vg_metadata_dir,
        cfg=cfg,
        split_name="train",
        vg_coco_overlap=vg_coco_overlap,
    )

    # Preprocess test split
    test_metadata = preprocess_split(
        vg_metadata=vg_metadata,
        nsd_coco_ids=nsd_coco_ids,
        vg_metadata_dir=vg_metadata_dir,
        cfg=cfg,
        split_name="test",
        vg_coco_overlap=vg_coco_overlap,
    )

    # Match data types for pushing to HF hub if needed
    if cfg.data.include_text:
        train_metadata = train_metadata.cast_column("coco_id", Value("int64"))  # type: ignore
        test_metadata = test_metadata.cast_column("coco_id", Value("int64"))  # type: ignore
        train_metadata = train_metadata.cast_column("sentences_raw", Value("string"))  # type: ignore
        test_metadata = test_metadata.cast_column("sentences_raw", Value("string"))  # type: ignore
        train_metadata = train_metadata.cast_column("flickr_id", Value("int64"))  # type: ignore
        test_metadata = test_metadata.cast_column("flickr_id", Value("int64"))  # type: ignore

    # Push both splits to the Hugging Face Hub
    dataset_dict = DatasetDict({"train": train_metadata, "test": test_metadata})
    dataset_dict.push_to_hub(
        repo_id=cfg.data.processed_hf_identifier,
    )


def derive_image_graphs(
    vg_objects_file: str,
    vg_relationships_file: str,
    vg_visual_verbs_file: str,
    vg_metadata_dir: Path,
    cfg: DictConfig,
    image_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Get the graph data of the VG + COCO overlap dataset for the given image ids.

    :param vg_objects_file: Path to the file where the Visual Genome objects json is stored.
    :type vg_objects_file: str
    :param vg_relationships_file: Path to the file where the Visual Genome relationship json is stored.
    :type vg_relationships_file: str
    :param vg_visual_verbs_file: Path to the file where the Visual VerbNet json is stored
    :type vg_visual_verbs_file: str
    :param vg_metadata_dir: Directory containing VG metadata files
    :type vg_metadata_dir: Path
    :param cfg: Configuration object containing patch sizes and target resolution
    :type cfg: DictConfig
    :param image_ids: Optional list of image ids to characterize the graph complexity for, defaults to None
    :type image_ids: Optional[List[str]]
    :return: Three dictionaries with the graph complexity measures (whole graph, actions, spatial rels) and image id
    :rtype: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]
    """
    # Load the object and relationship files from json
    vg_objects = load_dataset("json", data_files=str(vg_objects_file), split="train")
    vg_relationships = load_dataset("json", data_files=str(vg_relationships_file), split="train")

    # Load image metadata to get image dimensions
    vg_image_data_file = vg_metadata_dir / "image_data.json"
    vg_image_data = load_dataset("json", data_files=str(vg_image_data_file), split="train")
    # Create a mapping from image_id to image dimensions
    image_dims = {img["image_id"]: (img["width"], img["height"]) for img in vg_image_data}
    # Filter by image ids if given
    if image_ids:
        vg_objects = vg_objects.filter(lambda x: x["image_id"] in image_ids, num_proc=4)
        vg_relationships = vg_relationships.filter(
            lambda x: x["image_id"] in image_ids,
            num_proc=4,
        )
    # Load the Visual VerbNet file
    visual_verbs_data = load_dataset("json", data_files=str(vg_visual_verbs_file), split="train")
    visual_verbs = [entry["name"] for entry in visual_verbs_data["visual_actions"][0]]

    # Process each VG image/graph into a networkx graph
    graphs = {}
    action_graphs = {}
    spatial_graphs = {}
    # Store object information for patch index calculation
    object_data: dict = {}

    for obj, rel in tqdm(
        zip(vg_objects, vg_relationships),
        desc="Processing rels/objs as networkx graphs",
        total=len(vg_objects),
    ):
        image_id = obj["image_id"]

        # Get image dimensions for this image
        if image_id not in image_dims:
            logger.warning(f"No image dimensions found for image_id {image_id}, skipping")
            continue
        img_width, img_height = image_dims[image_id]

        # Store object data for patch index calculation
        object_data[image_id] = {}

        # Create the graph based on objects and relationships
        graph = nx.DiGraph()
        for o in obj["objects"]:
            object_id = o["object_id"]
            graph.add_node(object_id)

            # Calculate object center coordinates
            obj_x = o["x"] + o["w"] / 2
            obj_y = o["y"] + o["h"] / 2

            # Store object information including patch indices for different patch sizes
            patch_indices = {}
            for patch_size in cfg.data.patch_sizes:
                patch_idx = map_vg_to_clip_patch(
                    obj_x,
                    obj_y,
                    img_width,
                    img_height,
                    patch_size=patch_size,
                    target_resolution=cfg.data.target_resolution,
                )
                patch_indices[f"patch_{patch_size}"] = patch_idx

            object_data[image_id][object_id] = {
                "bbox": {"x": o["x"], "y": o["y"], "w": o["w"], "h": o["h"]},
                "patch_indices": patch_indices,
            }

        for r in rel["relationships"]:
            # If both subject and object are in obj["objects"], add the edge
            if r["subject"]["object_id"] in graph.nodes and r["object"]["object_id"] in graph.nodes:
                # Add the relationship as an edge with the relationship ID as an attribute
                graph.add_edge(
                    r["object"]["object_id"],
                    r["subject"]["object_id"],
                    rel_id=r["relationship_id"],
                )

        # Append the graph to the dict
        graphs[image_id] = graph
        # Filter relationships by visual actions
        action_rels = [
            r
            for r in rel["relationships"]
            if r["subject"]["object_id"] in graph.nodes
            and r["object"]["object_id"] in graph.nodes
            and len(r["object"]["synsets"]) > 0
            and len(r["subject"]["synsets"]) > 0
            and len(r["synsets"]) > 0
            and (check_if_living_being(r["object"]["synsets"][0]) or check_if_living_being(r["subject"]["synsets"][0]))
            and ".v." in r["synsets"][0]
            and r["synsets"][0].split(".")[0] in visual_verbs
        ]
        action_rel_ids = [r["relationship_id"] for r in action_rels]
        action_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get("rel_id") in action_rel_ids]
        # Create a new graph with the action edges and only nodes that have edges
        action_graph = nx.DiGraph(action_edges)
        # Remove isolated nodes (nodes with no edges)
        isolated_nodes = list(nx.isolates(action_graph))
        action_graph.remove_nodes_from(isolated_nodes)
        action_graphs[image_id] = action_graph

        # Do the same with spatial relations, i.e., if ".r." in the relationship synset
        spatial_rels = [
            r
            for r in rel["relationships"]
            if r["subject"]["object_id"] in graph.nodes
            and r["object"]["object_id"] in graph.nodes
            and len(r["synsets"]) > 0
            and ".r." in r["synsets"][0]
        ]
        spatial_rel_ids = [r["relationship_id"] for r in spatial_rels]
        spatial_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get("rel_id") in spatial_rel_ids]
        # Create a new graph with the spatial edges and only nodes that have edges
        spatial_graph = nx.DiGraph(spatial_edges)
        # Remove isolated nodes (nodes with no edges)
        isolated_nodes = list(nx.isolates(spatial_graph))
        spatial_graph.remove_nodes_from(isolated_nodes)
        spatial_graphs[image_id] = spatial_graph

    # Calculate the graphormer attributes
    graphs_graphormer = {}
    action_graphs_graphormer = {}
    spatial_graphs_graphormer = {}
    for (
        (graph_id, graph),
        (action_graph_id, action_graph),
        (spatial_graph_id, spatial_graph),
        (obj_meta_id, obj_data),
    ) in tqdm(
        zip(graphs.items(), action_graphs.items(), spatial_graphs.items(), object_data.items()),
        desc="Calculating graphormer attributes",
        total=len(graphs),
    ):
        assert graph_id == action_graph_id == spatial_graph_id == obj_meta_id, "IDs in wrong order"
        graphs_graphormer[graph_id] = calculate_graphormer_attributes(graph, obj_data, cfg.data.patch_sizes)
        action_graphs_graphormer[action_graph_id] = calculate_graphormer_attributes(
            action_graph, obj_data, cfg.data.patch_sizes
        )
        spatial_graphs_graphormer[spatial_graph_id] = calculate_graphormer_attributes(
            spatial_graph, obj_data, cfg.data.patch_sizes
        )

    return graphs_graphormer, action_graphs_graphormer, spatial_graphs_graphormer


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


def calculate_graphormer_attributes(
    graph: nx.Graph,
    object_data: Optional[Dict[int, Any]] = None,
    patch_sizes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Calculate the edge_index, num_nodes, and patch indices for a given networkx graph.

    :param graph: The input graph
    :type graph: nx.Graph
    :param object_data: Dictionary containing object information including patch indices
    :type object_data: Optional[Dict[int, Any]]
    :param patch_sizes: List of patch sizes to include in output
    :type patch_sizes: Optional[List[int]]
    :return: A dictionary containing edge_index, num_nodes, and patch indices
    :rtype: Dict[str, Any]
    """
    # Store original nodes before relabeling
    original_nodes = list(graph.nodes())

    # Create a mapping from original node IDs to sequential IDs starting from 0
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}

    # Relabel the nodes in the graph using the mapping
    relabeled_graph = nx.relabel_nodes(graph, node_mapping)

    # Convert edge_index to a 2 x n_edges format
    edge_index = [[u, v] for u, v in list(relabeled_graph.edges())]
    edge_index = list(map(list, zip(*edge_index)))  # Transpose to 2 x n_edges format
    # If there are no edges, create an empty edge_index
    if len(edge_index) == 0:
        edge_index = [[], []]

    # Get the number of nodes
    num_nodes = relabeled_graph.number_of_nodes()

    result = {"edge_index": edge_index, "num_nodes": num_nodes}

    # Add patch indices if object data is provided
    if object_data and patch_sizes:
        # Create patch indices arrays for each patch size
        for patch_size in patch_sizes:
            patch_key = f"patch_{patch_size}"
            patch_indices = []

            # Get patch indices for nodes in their original order (before relabel)
            # This maintains the same order as the relabeled node indices
            for node_id in original_nodes:
                if node_id in object_data and patch_key in object_data[node_id]["patch_indices"]:
                    patch_indices.append(object_data[node_id]["patch_indices"][patch_key])
                else:
                    # Use -1 as a placeholder for missing patch data to maintain array length
                    logger.warning(f"Missing patch data for node {node_id}, patch size {patch_size}, using -1")
                    patch_indices.append(-1)

            result[f"patch_indices_{patch_size}"] = patch_indices

    return result


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
    nlp = spacy.load(
        spacy_model,
        disable=["tok2vec", "attribute_ruler", "lemmatizer", "ner", "tagger"],
    )
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
