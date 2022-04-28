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

    dataset_stats = []

    for batch in tqdm(dataloader):

        stats = {
            'background': 0,
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0,
            'bicycle': 0,
            'scooter': 0,
            'person': 0,
            'rider': 0,
        }

        # Remove the two lines below and start analyzing :D
        # print("The keys in the batch are:", batch.keys())

        # print(f"image.shape={batch['image'].shape}")
        # print(f"boxes={batch['boxes']}")
        # print(f"labels={batch['labels']}")
        # print(f"width={batch['width']}")
        # print(f"height={batch['height']}")

        for idx, label in enumerate(batch['labels'].flatten()):
            # as coordinates are given relatively to dimensions this
            # calculates the relative size of bb compared to entire image
            bb = batch['boxes'][0][idx]
            size = (bb[2].item() - bb[0].item()) * (bb[3].item() - bb[1].item())
            
            # translate label index into label string
            label = cfg.label_map[label.item()]
            
            # append size to our stats dict
            stats[label] += size

        dataset_stats.append(stats)

    with open('dataset_stats.json', 'w') as f:
        json.dump(dataset_stats, f)
    

def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
