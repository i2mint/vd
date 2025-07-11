import pytest
from typing import (
    Callable,
    Any,
    Dict,
    List,
    Tuple,
    get_args,
    Sequence,
    NewType,
    Iterable,
    Literal,
)
from functools import partial
from collections.abc import MutableMapping

from dol import Pipe

from vd.dog import DOG, ADOG
from au.base import ComputationStatus
import time


# --- New Type Definitions (as specified by user) ---
Segment = NewType("Segment", str)
Embedding = NewType("Embedding", Sequence[float])
PlanarVector = Tuple[float, float]  # This is a direct tuple type
ClusterIndex = NewType("ClusterIndex", int)

Segments = Iterable[Segment]
Embeddings = Iterable[Embedding]
PlanarVectors = Iterable[PlanarVector]
ClusterIndices = Iterable[ClusterIndex]

# Callable types for operations
Embedder = Callable[[Segments], Embeddings]
Planarizer = Callable[[Embeddings], PlanarVectors]
Clusterer = Callable[[Embeddings], ClusterIndices]


# The `vectorize` utility function (as provided in the original problem description)
vectorize = lambda func: Pipe(partial(map, func), list)

# --- Test Data & Configuration (as provided in the user story test, adjusted) ---

# Define operation signatures (abstract function types and their I/O)
operation_signatures = {
    "embedder": Embedder,  # Callable[[Segments], Embeddings]
    "planarizer": Planarizer,  # Callable[[Embeddings], PlanarVectors]
    "clusterer": Clusterer,  # Callable[[Embeddings], ClusterIndices]
}

# Define data store configurations (what data types they hold, and initial data)
data_stores = {
    "segments": {
        "type": Segments,
        "store": {
            "segments_1": ["segment1", "segment2", "segment3"],
            "segments_2": ["segment4", "segment5"],
        },
    },
    "embeddings": {
        "type": Embeddings,
        "store": dict(),  # This store will hold Embeddings objects
    },
    "planar_vectors": {
        "type": PlanarVectors,
        "store": dict(),  # This store will hold PlanarVectors objects
    },
    "cluster_indices": {
        "type": ClusterIndices,
        "store": dict(),  # This store will hold ClusterIndices objects
    },
}

# Define concrete operation implementations (the actual functions)
operation_implementations = {
    "embedder": {
        "constant": lambda segments: vectorize(lambda s: [1, 2, 3])(segments),
        "segment_based": lambda segments: vectorize(lambda s: [len(s), 0.5, 0.5])(
            segments
        ),
    },
    "planarizer": {
        "constant": lambda embeddings: vectorize(lambda e: (e[0], e[1]))(embeddings),
        "embedding_based": lambda embeddings: vectorize(
            lambda e: [e[0] * 0.5, e[1] * 0.5]
        )(embeddings),
    },
    "clusterer": {
        # Note: The original Clusterer signature was Callable[[Embeddings, int], ClusterIndices]
        # but your new Clusterer type is Callable[[Embeddings], ClusterIndices].
        # Adjusted 'kmeans' to remove 'num_clusters' arg to match new signature.
        "kmeans": lambda embeddings: [0, 1]
        * (len(embeddings) // 2 + len(embeddings) % 2),  # Using 0, 1 for ClusterIndex
        "dbscan": lambda embeddings: [-1]
        * len(embeddings),  # Using -1 for noise (ClusterIndex)
    },
}


# --- Helper Function to Get Test Stores and Operations ---
def get_test_stores_and_ops(stores_type: Literal["ram", "local"] = "ram"):
    if stores_type == "ram":
        import copy

        data_stores = copy.deepcopy(data_stores_orig)
        op_impls = operation_implementations
    else:
        from imbed.imbed_project import get_mall  # pip install imbed

        mall = get_mall("dog_tests", get_project_mall=stores_type)
        data_stores = {
            "segments": {"type": Segments, "store": mall["segments"]},
            "embeddings": {"type": Embeddings, "store": mall["embeddings"]},
            "planar_vectors": {
                "type": PlanarVectors,
                "store": mall["planar_embeddings"],
            },
            "cluster_indices": {"type": ClusterIndices, "store": mall["clusters"]},
        }
        op_impls = operation_implementations
        # Print rootdir for each store
        print("\n[INFO] Local store root directories:")
        for name, store in data_stores.items():
            s = store["store"]
            rootdir = getattr(s, "rootdir", None)
            print(f"  {name}: {rootdir}")
        # Optionally clear the stores for a clean test
        for store in data_stores.values():
            keys = list(store["store"].keys())
            for k in keys:
                try:
                    del store["store"][k]
                except Exception:
                    pass
        # Add initial data for segments
        data_stores["segments"]["store"].update(
            {
                "segments_1": ["segment1", "segment2", "segment3"],
                "segments_2": ["segment4", "segment5"],
            }
        )
    return data_stores, op_impls


# Save the original data_stores for RAM mode
import copy

data_stores_orig = copy.deepcopy(data_stores)


# --- The User Story Test ---
# This test function defines the expected behavior of the DOG.
# It must run successfully against the implemented DOG class.
def test_dog_operations(stores_type: Literal["ram", "local"] = "ram"):
    data_stores, op_impls = get_test_stores_and_ops(stores_type)
    dog = DOG(
        operation_signatures=operation_signatures,
        data_stores=data_stores,
        operation_implementations=op_impls,
    )

    # 2. Inspect Data Stores
    # We want to confirm that all expected data stores are properly set up and accessible.
    print("\n--- Data Store Inspection ---")
    expected_data_stores = [
        "segments",
        "embeddings",
        "planar_vectors",
        "cluster_indices",
    ]
    assert sorted(list(dog.data_stores.keys())) == sorted(expected_data_stores)
    print(f"All expected data stores are present: {list(dog.data_stores.keys())}.")

    # We expect the 'segments' data store to contain its initial predefined data.
    assert "segments_1" in dog.data_stores["segments"]
    assert dog.data_stores["segments"]["segments_1"] == [
        "segment1",
        "segment2",
        "segment3",
    ]
    print("Initial 'segments_1' data verified.")

    # 3. Inspect Operation Implementations
    # We want to verify that all expected operation types (e.g., 'embedder') are registered
    # and that their concrete implementations are available.
    print("\n--- Operation Implementations Inspection ---")
    expected_operation_types = ["embedder", "planarizer", "clusterer"]
    assert sorted(list(dog.operation_implementations.keys())) == sorted(
        expected_operation_types
    )
    print(
        f"All expected operation types are registered: {list(dog.operation_implementations.keys())}."
    )

    # Specifically, check for the 'constant' and 'segment_based' implementations of the 'embedder'.
    assert "constant" in dog.operation_implementations["embedder"]
    assert "segment_based" in dog.operation_implementations["embedder"]
    print("Embedder operation implementations ('constant', 'segment_based') verified.")

    # 4. Perform CRUD Operations on Data Stores
    # The DOG should allow standard Create, Read, Update, and Delete operations on its managed data stores.
    print("\n--- CRUD Operations on Data Stores ---")

    # Create: Add new data to the 'segments' store.
    dog.data_stores["segments"]["segments_3"] = ["segment6", "segment7"]
    assert "segments_3" in dog.data_stores["segments"]
    assert dog.data_stores["segments"]["segments_3"] == ["segment6", "segment7"]
    print("New 'segments_3' data added successfully.")

    # Update: Modify existing data within the 'segments' store.
    dog.data_stores["segments"]["segments_1"] = [
        "updated_segment_A",
        "updated_segment_B",
    ]
    assert dog.data_stores["segments"]["segments_1"] == [
        "updated_segment_A",
        "updated_segment_B",
    ]
    print("Existing 'segments_1' data updated successfully.")

    # Read: Retrieve data from the 'segments' store.
    retrieved_segments = dog.data_stores["segments"]["segments_2"]
    assert retrieved_segments == ["segment4", "segment5"]
    print("Data for 'segments_2' retrieved successfully.")

    # Delete: Remove data from the 'segments' store.
    del dog.data_stores["segments"]["segments_3"]
    assert "segments_3" not in dog.data_stores["segments"]
    print("Data for 'segments_3' deleted successfully.")

    # 5. Execute Operations and Manage Outputs
    # This section demonstrates the core functionality of the DOG: calling operations
    # and automatically managing their outputs within the appropriate data stores.
    print("\n--- Operation Execution and Output Management ---")

    # A. Call 'embedder' (constant) operation on 'segments_1' data.
    # The output (embeddings) should be automatically directed to the 'embeddings' data store.
    segments_for_embedding = dog.data_stores["segments"][
        "segments_1"
    ]  # Use the recently updated 'segments_1'
    output_store_key_embed, output_val_key_embed = dog.call(
        dog.operation_implementations["embedder"]["constant"], segments_for_embedding
    )

    # Assert that the output was indeed stored in 'embeddings' and verify its content.
    assert output_store_key_embed == "embeddings"
    retrieved_embeddings = dog.data_stores[output_store_key_embed][output_val_key_embed]
    # Since 'segments_1' has 2 items, the constant embedder produces 2 outputs.
    assert retrieved_embeddings == [[1, 2, 3], [1, 2, 3]]
    print(
        f"Embedder 'constant' executed. Output stored at '{output_store_key_embed}' with key '{output_val_key_embed}'."
    )
    print(f"Retrieved embeddings: {retrieved_embeddings}.")

    # B. Call 'planarizer' (embedding_based) operation on the retrieved embeddings.
    # This demonstrates chaining operations where the output of one becomes the input for another.
    output_store_key_planar, output_val_key_planar = dog.call(
        dog.operation_implementations["planarizer"]["embedding_based"],
        retrieved_embeddings,  # Use the actual embeddings obtained from the previous step
    )

    # Assert that the output was stored in 'planar_vectors' and verify its content.
    assert output_store_key_planar == "planar_vectors"
    retrieved_planar_vectors = dog.data_stores[output_store_key_planar][
        output_val_key_planar
    ]
    # Based on embeddings [[1,2,3],[1,2,3]], each [1,2,3] transforms to [1*0.5, 2*0.5] = [0.5, 1.0].
    assert retrieved_planar_vectors == [[0.5, 1.0], [0.5, 1.0]]
    print(
        f"Planarizer 'embedding_based' executed. Output stored at '{output_store_key_planar}' with key '{output_val_key_planar}'."
    )
    print(f"Retrieved planar vectors: {retrieved_planar_vectors}.")

    # C. Call 'clusterer' (kmeans) operation using the generated embeddings.
    # Note: The 'Clusterer' signature was updated, so no 'num_clusters' direct input here.
    output_store_key_cluster, output_val_key_cluster = dog.call(
        dog.operation_implementations["clusterer"]["kmeans"],
        retrieved_embeddings,  # Embeddings from previous step
    )

    # Assert that the output was stored in 'cluster_indices' and verify its content.
    assert output_store_key_cluster == "cluster_indices"
    retrieved_cluster_indices = dog.data_stores[output_store_key_cluster][
        output_val_key_cluster
    ]
    # For 2 segments (inputs to embedder), expected output is [0, 1] based on adjusted lambda.
    assert retrieved_cluster_indices == [0, 1]
    print(
        f"Clusterer 'kmeans' executed. Output stored at '{output_store_key_cluster}' with key '{output_val_key_cluster}'."
    )
    print(f"Retrieved cluster indices: {retrieved_cluster_indices}.")

    print("\n--- All DOG operations tested successfully! ---")


# Test ADOG operations
def test_adog_operations(stores_type: Literal["ram", "local"] = "ram"):
    """
    Test the ADOG async operation graph. This mirrors the DOG test but checks async result storage.
    """
    # Use a fresh data_stores dict for each test
    data_stores, op_impls = get_test_stores_and_ops(stores_type)
    adog = ADOG(
        operation_signatures=operation_signatures,
        data_stores=data_stores,
        operation_implementations=op_impls,
    )

    # CRUD: Add and update segments
    adog.data_stores["segments"]["segments_3"] = ["segment6", "segment7"]
    adog.data_stores["segments"]["segments_1"] = [
        "updated_segment_A",
        "updated_segment_B",
    ]
    del adog.data_stores["segments"]["segments_3"]

    # --- Async Operation Execution ---
    segments_for_embedding = adog.data_stores["segments"]["segments_1"]
    output_store_key_embed, output_val_key_embed = adog.call(
        adog.operation_implementations["embedder"]["constant"], segments_for_embedding
    )
    # Wait for async result to appear in the store
    for _ in range(50):  # up to 5 seconds
        if output_val_key_embed in adog.data_stores[output_store_key_embed]:
            computation_result = adog.data_stores[output_store_key_embed][
                output_val_key_embed
            ]
            if computation_result.status == ComputationStatus.COMPLETED:
                break
            elif computation_result.status == ComputationStatus.FAILED:
                raise RuntimeError(
                    f"ADOG embedder computation failed: {computation_result.error}"
                )
        time.sleep(0.1)
    else:
        raise TimeoutError("ADOG embedder async result did not appear in time")

    computation_result = adog.data_stores[output_store_key_embed][output_val_key_embed]
    retrieved_embeddings = computation_result.value
    assert retrieved_embeddings == [[1, 2, 3], [1, 2, 3]]

    # Planarizer async
    output_store_key_planar, output_val_key_planar = adog.call(
        adog.operation_implementations["planarizer"]["embedding_based"],
        retrieved_embeddings,
    )
    for _ in range(50):
        if output_val_key_planar in adog.data_stores[output_store_key_planar]:
            computation_result = adog.data_stores[output_store_key_planar][
                output_val_key_planar
            ]
            if computation_result.status == ComputationStatus.COMPLETED:
                break
        time.sleep(0.1)
    else:
        raise TimeoutError("ADOG planarizer async result did not appear in time")

    computation_result = adog.data_stores[output_store_key_planar][
        output_val_key_planar
    ]
    retrieved_planar_vectors = computation_result.value
    assert retrieved_planar_vectors == [[0.5, 1.0], [0.5, 1.0]]

    # Clusterer async
    output_store_key_cluster, output_val_key_cluster = adog.call(
        adog.operation_implementations["clusterer"]["kmeans"],
        retrieved_embeddings,
    )
    for _ in range(50):
        if output_val_key_cluster in adog.data_stores[output_store_key_cluster]:
            computation_result = adog.data_stores[output_store_key_cluster][
                output_val_key_cluster
            ]
            if computation_result.status == ComputationStatus.COMPLETED:
                break
        time.sleep(0.1)
    else:
        raise TimeoutError("ADOG clusterer async result did not appear in time")

    computation_result = adog.data_stores[output_store_key_cluster][
        output_val_key_cluster
    ]
    retrieved_cluster_indices = computation_result.value
    assert retrieved_cluster_indices == [0, 1]

    print("\n--- All ADOG async operations tested successfully! ---")


def test_dog_sourced_operation(stores_type: Literal["ram", "local"] = "ram"):
    """
    Test DOG with sourced_argnames: when an argument name is in sourced_argnames,
    and the value passed is a key, the value is fetched from the corresponding store.
    """
    data_stores, op_impls = get_test_stores_and_ops(stores_type)
    sourced_argnames = {"segments": "segments", "embeddings": "embeddings"}
    dog = DOG(
        operation_signatures=operation_signatures,
        data_stores=data_stores,
        operation_implementations=op_impls,
        sourced_argnames=sourced_argnames,
    )
    # Store a new segments list under a key
    dog.data_stores["segments"]["my_segments"] = ["a", "b", "c"]
    # Call embedder with a key, not the value
    output_store, output_key = dog.call(
        dog.operation_implementations["embedder"]["constant"], segments="my_segments"
    )
    result = dog.data_stores[output_store][output_key]
    assert result == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    # Now test chaining: planarizer with embeddings key
    dog.data_stores["embeddings"]["my_embeds"] = [[1, 2, 3], [4, 5, 6]]
    output_store2, output_key2 = dog.call(
        dog.operation_implementations["planarizer"]["constant"], embeddings="my_embeds"
    )
    result2 = dog.data_stores[output_store2][output_key2]
    assert list(map(list, result2)) == [[1, 2], [4, 5]]
    print("test_dog_sourced_operation passed.")


def run_tests(stores_type="ram"):
    """
    Run all tests defined in this module.
    This is useful for running tests in a script or interactive environment.
    """
    if stores_type is not None:
        test_dog_operations(stores_type)
        test_adog_operations(stores_type)
        test_dog_sourced_operation(stores_type)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DOG tests.")
    parser.add_argument(
        "--store_type",
        type=str,
        choices=["ram", "local"],
        default="ram",
        help="Specify the store type to use ('ram' or 'local').",
    )
    args = parser.parse_args()
    run_tests(args.store_type)
