from __future__ import print_function, absolute_import
import argparse
import sys
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

import models
from opts import get_opts
from util import data_manager
from util.dataset_loader import ImageDataset
from util.eval_metrics import make_results


EPSILON = 8.0 / 255.0
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXPECTED_CLEAN = {
    'mAP': 0.6113,
    'Rank-1': 0.8088,
    'Rank-5': 0.9210,
    'Rank-10': 0.9480,
}
BASELINE_TOLERANCE = 0.002


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Duke-trained AP-Attack against Market1501 IDE.'
    )
    parser.add_argument(
        '--data-root', '--root', dest='data_root', type=str,
        default='/home/lzf/ldx/datasets'
    )
    parser.add_argument(
        '--victim-checkpoint', '--victim_weight', dest='victim_checkpoint', type=str,
        default=(
            '/home/lzf/ldx/checkpoints/DeepMisRanking/IDE/market1501/'
            'market1501.pth.tar'
        ),
    )
    parser.add_argument(
        '--apattack-root', '--apattack_root', dest='apattack_root', type=str,
        default='/home/lzf/ldx/projects/AP-Attack'
    )
    parser.add_argument(
        '--generator-checkpoint', '--generator_weight',
        dest='generator_checkpoint', type=str,
        default=(
            '/home/lzf/ldx/checkpoints/AP-Attack/generator/'
            'apattack_duke_stage2_best_ep60.pth.tar'
        ),
    )
    parser.add_argument('--batch-size', '--test_batch', dest='batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def extract_ide_feature(model, images):
    return model(images, False)[0]


def generate_adversarial_images(generator, images):
    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)

    clean_pixel = (images * std + mean).clamp(0, 1)
    generator_input = (clean_pixel - mean) / std
    delta_raw = generator(generator_input)
    delta_pixel = delta_raw.clamp(-EPSILON, EPSILON)
    adv_pixel = (clean_pixel + delta_pixel).clamp(0, 1)
    adv_images = (adv_pixel - mean) / std
    actual_delta = adv_pixel - clean_pixel
    return adv_images, actual_delta


def load_generator(apattack_root, generator_checkpoint, device):
    sys.path.insert(0, apattack_root)
    from advers.GD import Generator

    generator = Generator(3, 3, 32, norm='bn', beta=0.1)
    checkpoint = torch.load(generator_checkpoint, map_location='cpu')
    state_dict = (
        checkpoint['state_dict']
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint
        else checkpoint
    )
    state_dict = OrderedDict(
        (key[7:] if key.startswith('module.') else key, value)
        for key, value in state_dict.items()
    )
    generator.load_state_dict(state_dict, strict=True)
    generator.to(device)
    generator.eval()
    return generator


def extract_query_features(loader, model, generator, device):
    clean_features, adv_features = [], []
    pids, camids = [], []
    attacked_query_count = 0
    max_actual_delta = 0.0
    total_abs_delta = 0.0
    total_delta_values = 0

    for batch_idx, (images, batch_pids, batch_camids, _) in enumerate(loader):
        images = images.to(device)
        clean_features.append(extract_ide_feature(model, images).cpu())

        adv_images, actual_delta = generate_adversarial_images(generator, images)
        adv_features.append(extract_ide_feature(model, adv_images).cpu())

        max_actual_delta = max(max_actual_delta, actual_delta.abs().max().item())
        total_abs_delta += actual_delta.abs().sum().item()
        total_delta_values += actual_delta.numel()
        attacked_query_count += images.size(0)
        pids.extend(batch_pids.cpu().numpy())
        camids.extend(batch_camids.cpu().numpy())
        print('Processed query batch {}/{}'.format(batch_idx + 1, len(loader)))

    assert max_actual_delta <= EPSILON + 1e-6
    return (
        torch.cat(clean_features, 0),
        torch.cat(adv_features, 0),
        np.asarray(pids),
        np.asarray(camids),
        attacked_query_count,
        max_actual_delta,
        total_abs_delta / total_delta_values,
    )


def extract_gallery_features(loader, model, device):
    features, pids, camids = [], [], []
    for batch_idx, (images, batch_pids, batch_camids, _) in enumerate(loader):
        images = images.to(device)
        features.append(extract_ide_feature(model, images).cpu())
        pids.extend(batch_pids.cpu().numpy())
        camids.extend(batch_camids.cpu().numpy())
        print('Processed gallery batch {}/{}'.format(batch_idx + 1, len(loader)))
    return torch.cat(features, 0), np.asarray(pids), np.asarray(camids)


def print_results(title, cmc, mAP):
    print('\n=== {} ==='.format(title))
    print('mAP: {:.2%}'.format(mAP))
    print('Rank-1: {:.2%}'.format(cmc[0]))
    print('Rank-5: {:.2%}'.format(cmc[4]))
    print('Rank-10: {:.2%}'.format(cmc[9]))


def clean_baseline_matches(cmc, mAP):
    actual = {
        'mAP': mAP,
        'Rank-1': cmc[0],
        'Rank-5': cmc[4],
        'Rank-10': cmc[9],
    }
    return all(
        abs(actual[name] - expected) <= BASELINE_TOLERANCE
        for name, expected in EXPECTED_CLEAN.items()
    )


def print_perturbation_sanity(attacked_query_count, max_actual_delta, mean_actual_delta):
    print('\n=== PERTURBATION SANITY ===')
    print('attacked query count: {}'.format(attacked_query_count))
    print('epsilon: {:.8f}'.format(EPSILON))
    print('max actual |delta|: {:.8f}'.format(max_actual_delta))
    print('mean actual |delta|: {:.8f}'.format(mean_actual_delta))


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')

    opt = get_opts('ide')
    workers = opt['workers'] if args.workers is None else args.workers
    dataset = data_manager.init_img_dataset(
        root=args.data_root,
        name='market1501',
        split_id=0,
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
    )
    query_loader = DataLoader(
        ImageDataset(dataset.query, transform=opt['transform_test']),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == 'cuda',
        drop_last=False,
    )
    gallery_loader = DataLoader(
        ImageDataset(dataset.gallery, transform=opt['transform_test']),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == 'cuda',
        drop_last=False,
    )

    model = models.init_model(
        name='ide',
        pre_dir=args.victim_checkpoint,
        num_classes=dataset.num_train_pids,
        pretrained=False,
    )
    model.to(device)
    model.eval()
    generator = load_generator(args.apattack_root, args.generator_checkpoint, device)

    with torch.no_grad():
        (
            clean_qf,
            adv_qf,
            q_pids,
            q_camids,
            attacked_query_count,
            max_actual_delta,
            mean_actual_delta,
        ) = extract_query_features(query_loader, model, generator, device)
        gf, g_pids, g_camids = extract_gallery_features(gallery_loader, model, device)

    assert attacked_query_count == len(dataset.query)
    assert attacked_query_count == 3368
    _, clean_cmc, clean_mAP = make_results(
        clean_qf,
        gf,
        [],
        [],
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        targetmodel='ide',
        ak_typ=-1,
    )
    print_results('CLEAN RESULTS', clean_cmc, clean_mAP)
    if not clean_baseline_matches(clean_cmc, clean_mAP):
        print_perturbation_sanity(
            attacked_query_count, max_actual_delta, mean_actual_delta
        )
        print(
            '\nClean baseline differs from the expected IDE logits protocol. '
            'Do not interpret adversarial results until this is resolved.'
        )
        return

    _, adv_cmc, adv_mAP = make_results(
        adv_qf,
        gf,
        [],
        [],
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        targetmodel='ide',
        ak_typ=-1,
    )
    print_results('ADVERSARIAL RESULTS', adv_cmc, adv_mAP)
    absolute_mAP_drop = clean_mAP - adv_mAP
    relative_mAP_drop = absolute_mAP_drop / clean_mAP * 100.0
    print('\n=== ATTACK SUMMARY ===')
    print('attacked query count: {}'.format(attacked_query_count))
    print('clean mAP: {:.2%}'.format(clean_mAP))
    print('adv mAP: {:.2%}'.format(adv_mAP))
    print('absolute mAP drop: {:.2f} percentage points'.format(absolute_mAP_drop * 100.0))
    print('relative mAP drop: {:.2f}%'.format(relative_mAP_drop))
    print_perturbation_sanity(
        attacked_query_count, max_actual_delta, mean_actual_delta
    )


if __name__ == '__main__':
    main()
