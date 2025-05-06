import json
import os

import hydra
import matplotlib.pyplot as plt
import networkx as nx
from omegaconf import DictConfig

from nsd_compositionality.data.preprocess_vg_for_graphormer import check_if_living_being


def visualize_scene_graph_with_networkx(graph, output_dir, image_file, filtered=False):
    """Creates a visualization of the scene graph using networkx.

    Args:
        graph: A scene graph object.
        output_dir: Directory to save the visualization files.
        image_file: The image file to save the graph visualization.
        filtered: Whether to visualize only filtered relationships and objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_file_path = os.path.join(output_dir, image_file)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes (objects)
    objects = graph["filtered_objects"] if filtered else graph["objects"]
    for obj in objects:
        G.add_node(obj["name"])

    # Add edges (relationships)
    relationships = graph["filtered_relationships"] if filtered else graph["relationships"]
    for rel in relationships:
        G.add_edge(rel["subject"], rel["object"], label=rel["predicate"])

    # Draw the graph
    pos = nx.spring_layout(G)  # Position nodes using a spring layout
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue", font_size=10, font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Save the graph as an image
    plt.savefig(image_file_path)
    plt.close()


def extract_graph_from_relationships(vg_id, relationships_file, filter_living_beings=False):
    """Extracts a scene graph from the relationships.json file using a VG ID.

    Args:
        vg_id: The Visual Genome ID of the scene.
        relationships_file: Path to the relationships JSON file.
        filter_living_beings: Whether to filter relationships involving living beings.

    Returns:
        A scene graph object.
    """
    # Load relationships data
    with open(relationships_file, "r") as f:
        relationships_data = json.load(f)

    # Find the relationships for the given VG ID
    for entry in relationships_data:
        if entry.get("image_id") == vg_id:
            objects = []
            relationships = []
            filtered_objects = []
            filtered_relationships = []

            # Extract objects and relationships
            for rel in entry["relationships"]:
                subject = rel["subject"]
                obj = rel["object"]

                # Add unique objects
                if subject not in objects:
                    objects.append(subject)
                if obj not in objects:
                    objects.append(obj)

                # Add relationships
                relationships.append(
                    {
                        "predicate": rel["predicate"],
                        "subject": subject["name"],
                        "object": obj["name"],
                    }
                )

                # Filter relationships involving living beings
                if filter_living_beings:
                    if (len(subject["synsets"]) > 0 and check_if_living_being(subject["synsets"][0])) or (
                        len(obj["synsets"]) > 0 and check_if_living_being(obj["synsets"][0])
                    ):
                        if subject not in filtered_objects:
                            filtered_objects.append(subject)
                        if obj not in filtered_objects:
                            filtered_objects.append(obj)
                        filtered_relationships.append(
                            {
                                "predicate": rel["predicate"],
                                "subject": subject["name"],
                                "object": obj["name"],
                            }
                        )

            return {
                "objects": objects,
                "relationships": relationships,
                "filtered_objects": filtered_objects,
                "filtered_relationships": filtered_relationships,
            }

    raise ValueError(f"No relationships found for VG ID {vg_id}")


@hydra.main(config_path="../../../configs/visualization", config_name="scene_graphs")
def main(cfg: DictConfig):
    """Main function to visualize a scene graph using Hydra configuration."""
    vg_id = cfg.vg_id
    relationships_file = cfg.relationships_file
    output_dir = cfg.output_dir
    image_file = cfg.image_file
    filtered = cfg.filtered  # Whether to visualize filtered relationships and objects

    # Extract the scene graph
    graph = extract_graph_from_relationships(vg_id, relationships_file, filter_living_beings=filtered)

    # Visualize the scene graph
    visualize_scene_graph_with_networkx(graph, output_dir, image_file, filtered=filtered)


if __name__ == "__main__":
    main()
