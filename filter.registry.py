from filters import range_filter

# Dictionary mapping filter names to their respective functions
FILTERS = {
    "range": range_filter.apply
}

def apply_filters(config, point_cloud):
    """Applies all filters from config to the point cloud.

    Args:
        config (dict): Configuration settings.
        point_cloud (dict): Point cloud data.

    Returns:
        np.ndarray: Global mask for filtering.
    """
    masks = []

    for filter_name, params in config["filtering"].items():
        if filter_name in FILTERS:
            mask = FILTERS[filter_name](params, point_cloud)
            masks.append(mask)

    # Combine all filter masks using logical AND
    global_mask = masks[0]
    for mask in masks[1:]:
        global_mask &= mask

    return global_mask
