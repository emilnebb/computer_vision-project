from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import json


def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_something(dataloader, cfg):

    rel_bb_sizes = []
    aspect_ratios = []

    bb_dimensions = []

    # counting # occurrences and sum of bb sizes
    total_class_counters = {
        'background': (0, 0.),
        'car': (0, 0.),
        'truck': (0, 0.),
        'bus': (0, 0.),
        'motorcycle': (0, 0.),
        'bicycle': (0, 0.),
        'scooter': (0, 0.),
        'person': (0, 0.),
        'rider': (0, 0.),
    }


    for i, batch in tqdm(enumerate(dataloader)):

        rel_bb_size = {
            'background': 0,
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0,
            'bicycle': 0,
            'scooter': 0,
            'person': 0,
            'rider': 0,
            'sum': 0
        }

        # Remove the two lines below and start analyzing :D
        # print("The keys in the batch are:", batch.keys())

        # print(f"image.shape={batch['image'].shape}")
        # print(f"boxes={batch['boxes']}")
        # print(f"labels={batch['labels']}")
        # print(f"width={batch['width']}")
        # print(f"height={batch['height']}")

        for idx, label in enumerate(batch['labels'].flatten()):

            # translate label index into label string
            label = cfg.label_map[label.item()]

            rel_x1, rel_x2 = batch['boxes'][0][idx][0].item(), batch['boxes'][0][idx][2].item()
            rel_y1, rel_y2 = batch['boxes'][0][idx][1].item(), batch['boxes'][0][idx][3].item()

            # as coordinates are given relatively to dimensions this
            # calculates the relative size of bb compared to entire image
            rel_width = rel_x2 - rel_x1
            rel_height = rel_y2 - rel_y1

            abs_width = rel_width * batch['width'].item()
            abs_height = rel_height * batch['height'].item()

            bb_dimensions.append((abs_width, abs_height, label))

            current = total_class_counters[label]
            total_class_counters[label] = (current[0] + 1, current[1] + rel_width * rel_height)

            # abs_width = rel_width * batch['width']


            rel_bb_size[label] += rel_width * rel_height
            rel_bb_size['sum'] += rel_width * rel_height
            
            # we use scalar as aspect ratio (height/width)
            # because most of the bb boxes have greater hight than width
            aspect_ratios.append(rel_height / rel_width)

        rel_bb_sizes.append(rel_bb_size)

    with open('dataset_exploration/bb_dimensions.json', 'w') as f:
        json.dump(bb_dimensions, f)
        
    with open('dataset_exploration/total_class_counters.json', 'w') as f:
        json.dump(total_class_counters, f)

    with open('dataset_exploration/bb_sizes.json', 'w') as f:
        json.dump(rel_bb_sizes, f)
        
    with open('dataset_exploration/aspect_ratios.json', 'w') as f:
        json.dump(aspect_ratios, f)
    

def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
