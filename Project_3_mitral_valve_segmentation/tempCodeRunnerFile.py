concatenated_masks = {}
    for name, mask in tqdm(zip(test_names, binary_masks), total=len(test_names), desc="Concatenating masks"):
        if name not in concatenated_masks:
            concatenated_masks[name] = mask
        else:
            concatenated_masks[name] = np.concatenate((concatenated_masks[name], mask), axis=0)

    for name, mask in tqdm(concatenated_masks.items(), desc="Generating submission data"):
        flattened_mask = mask.flatten()  # Flatten the concatenated binary mask
        start_indices, lengths = get_sequences(flattened_mask)  # Extract sequences
        
        for i, (start, length) in enumerate(zip(start_indices, lengths)):
            ids.append(f"{name}_{i}")  # Format ID as name_i
            values.append(f"[{start}, {length}]")  # Format value as [flattenedIdx, len]
