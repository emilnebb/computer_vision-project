from numpy import sqrt

def calculate_anchor_boxes():
    side = 64
    min_sizes=[30, 60, 111, 162, 213, 264, 315]
    aspect_ratios=[2, 3]

    for idx, min_size in enumerate(min_sizes):
        print(f"min_size={min_size}")
        print((sqrt(side) * min_size, sqrt(side) * min_size))

        if idx+1 < len(min_sizes):
            print((sqrt(side) * sqrt(min_size * min_sizes[idx+1]), sqrt(side) * sqrt(min_size * min_sizes[idx+1])))

        for aspect_ratio in aspect_ratios:
            print(f"aspect_ratio={aspect_ratio}")
            w, h = min_size * sqrt(aspect_ratio), min_size / sqrt(aspect_ratio)
            if w < 300 and h < 300:
                print((w, h))

            w, h = min_size / sqrt(aspect_ratio), min_size * sqrt(aspect_ratio)
            if w < 300 and h < 300:
                print((w, h))


calculate_anchor_boxes()

