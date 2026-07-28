# clean_bee_data.py
"""
Remove technical genomic data from bee JSON files,
keeping only the facts that are useful for RAG queries.
Also makes common_name the primary field.
"""

import json
import os
from typing import List, Dict


def clean_bee_facts(item: Dict) -> Dict:
    """
    Remove technical/genomic facts, keep bee-relevant information,
    and make common_name the primary field.
    """
    original_facts = item.get("facts", [])
    metadata = item.get("metadata", {})

    # Extract common name from metadata or facts
    common_name = None

    # Check if common_name exists in metadata
    if "common_name" in metadata and metadata["common_name"]:
        common_name = metadata["common_name"]

    # If not in metadata, check the facts
    if not common_name:
        for fact in original_facts:
            if fact.startswith("Common name:"):
                common_name = fact.replace("Common name:", "").strip()
                break

    # If still no common name, use the species name as fallback
    if not common_name:
        common_name = item.get("species", "Unknown Bee")

    # Get the species name
    species_name = item.get("species", "")

    # Define what to KEEP (bee-relevant information)
    keep_patterns = [
        "Common name:",
        "Species:",
        "Order:",
        "Family:",
        "Genus:",
        "Description:",
        "Distribution:",
        "Habitat:",
        "Behavior:",
        "Threats:",
        "Conservation status:",
        "Tissue samples:",
        "Geographic locations:",
        "Data source:",
        "Reference:",
    ]

    # Define what to REMOVE (technical/genomic data)
    remove_patterns = [
        "Genome size:",
        "GC content:",
        "Number of proteins:",
        "Number of scaffolds:",
        "N50:",
        "BUSCO score:",
        "OMArk completeness:",
        "Swiss-Prot annotations:",
        "GO annotations:",
        "KEGG annotations:",
        "PFAM annotations:",
        "miRNAs:",
        "lncRNAs:",
        "Transcripts available:",
        "Accession number:",
        "Article:",
    ]

    cleaned_facts = []

    for fact in original_facts:
        should_keep = True

        # Check if this fact should be removed
        for pattern in remove_patterns:
            if fact.startswith(pattern):
                should_keep = False
                break

        # Remove very short facts
        if should_keep and len(fact.split()) < 3:
            should_keep = False

        if should_keep:
            cleaned_facts.append(fact)

    # ============ REORDER: Common name FIRST ============
    # Build new fact list with common name at the top
    reordered_facts = []

    # 1. Common name first
    reordered_facts.append(f"Common name: {common_name}")

    # 2. Species name (as "Scientific name" for clarity)
    reordered_facts.append(f"Scientific name: {species_name}")

    # 3. Then taxonomy
    for fact in cleaned_facts:
        if fact.startswith("Order:") or fact.startswith("Family:") or fact.startswith("Genus:"):
            reordered_facts.append(fact)

    # 4. Then description
    for fact in cleaned_facts:
        if fact.startswith("Description:"):
            reordered_facts.append(fact)

    # 5. Then everything else (except common name and species which we already added)
    for fact in cleaned_facts:
        if not fact.startswith("Common name:") and not fact.startswith("Species:"):
            if fact not in reordered_facts:
                reordered_facts.append(fact)

    # Update the item
    item["common_name"] = common_name  # Add common_name as a top-level field
    item["species"] = species_name  # Keep species but now secondary
    item["facts"] = reordered_facts

    # Update metadata to reflect the change
    if "common_name" not in metadata:
        metadata["common_name"] = common_name

    return item


def clean_bee_data_file(input_file: str, output_file: str):
    """
    Load a bee data JSON file, clean it, and save to a new file.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []

    for item in data:
        cleaned_item = clean_bee_facts(item)
        cleaned_data.append(cleaned_item)

    # Save cleaned data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2)

    # Print statistics
    original_count = sum(len(item.get("facts", [])) for item in data)
    cleaned_count = sum(len(item.get("facts", [])) for item in cleaned_data)

    print(f"✅ Original facts: {original_count}")
    print(f"✅ Cleaned facts: {cleaned_count}")
    print(f"✅ Removed: {original_count - cleaned_count} facts")
    print(f"✅ Saved to: {output_file}")

    # Show first example
    if cleaned_data:
        print("\n📝 Example cleaned item:")
        print(f"   Common name: {cleaned_data[0].get('common_name', 'N/A')}")
        print(f"   Scientific name: {cleaned_data[0].get('species', 'N/A')}")
        print(f"   First 3 facts:")
        for fact in cleaned_data[0].get("facts", [])[:3]:
            print(f"     - {fact}")


# ============ RUN IT ============
if __name__ == "__main__":
    input_file = "C:/final_project_starter/data/sample_docs/bee_data_cleaned.json"
    output_file = "/data/sample_docs/bee_data_clean.json"

    clean_bee_data_file(input_file, output_file)