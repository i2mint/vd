"""
Demo script showing the backend comparison and recommendation features of vd.

This demonstrates how users can:
- Compare different backends
- Get backend recommendations based on requirements
- View backend characteristics and use cases
"""

import vd


def main():
    print("=" * 80)
    print("VD Backend Comparison and Recommendation Demo")
    print("=" * 80)
    print()

    # 1. Compare available backends
    print("1. Comparing Memory and ChromaDB backends:")
    print("-" * 80)
    vd.print_comparison(['memory', 'chroma'])

    # 2. Get recommendation for small dataset
    print("\n2. Get recommendation for small dataset (prototyping):")
    print("-" * 80)
    vd.print_recommendation(
        dataset_size='small', persistence_required=False, budget='free'
    )

    # 3. Get recommendation for medium dataset with persistence
    print("\n3. Get recommendation for medium dataset with persistence:")
    print("-" * 80)
    vd.print_recommendation(
        dataset_size='medium', persistence_required=True, budget='free'
    )

    # 4. Get recommendation for large cloud deployment
    print("\n4. Get recommendation for large-scale cloud deployment:")
    print("-" * 80)
    vd.print_recommendation(
        dataset_size='large',
        persistence_required=True,
        cloud_required=True,
        budget='medium',
        performance_priority='scalability',
    )

    # 5. Get recommendation for very large enterprise deployment
    print("\n5. Get recommendation for very large enterprise deployment:")
    print("-" * 80)
    vd.print_recommendation(
        dataset_size='very_large',
        persistence_required=True,
        cloud_required=False,
        budget='high',
        performance_priority='speed',
    )

    # 6. Show backend characteristics
    print("\n6. Backend characteristics and use cases:")
    print("-" * 80)
    chars = vd.get_backend_characteristics()

    for backend, info in list(chars.items())[:3]:  # Show first 3
        print(f"\n{backend.upper()}:")
        print(f"  Strengths:")
        for strength in info['strengths']:
            print(f"    • {strength}")
        print(f"  Best for:")
        for use_case in info['use_cases']:
            print(f"    • {use_case}")

    # 7. Programmatic comparison
    print("\n\n7. Programmatic backend comparison:")
    print("-" * 80)
    comparison = vd.compare_backends(['memory', 'chroma', 'pinecone'])
    print(f"Compared {len(comparison)} backends:")
    for backend, data in comparison.items():
        status = "available" if data['available'] else "not available"
        print(f"  • {backend}: {status}")
        if 'scalability' in data:
            print(f"    Scalability: {data['scalability']}")
        if 'best_for' in data:
            print(f"    Best for: {data['best_for']}")

    # 8. Get recommendation programmatically
    print("\n\n8. Programmatic recommendation:")
    print("-" * 80)
    rec = vd.recommend_backend(
        dataset_size='medium', persistence_required=True, budget='free'
    )
    print(f"Primary recommendation: {rec['primary']}")
    print(f"Alternatives: {', '.join(rec['alternatives'][:3])}")
    print(f"Reasoning:")
    for reason in rec['reasoning']:
        print(f"  • {reason}")

    print("\n" + "=" * 80)
    print("Comparison Features:")
    print("  • Compare multiple backends side-by-side")
    print("  • Get recommendations based on requirements")
    print("  • View detailed characteristics and use cases")
    print("  • Programmatic or formatted output")
    print("  • Budget, scale, and performance considerations")
    print("=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
