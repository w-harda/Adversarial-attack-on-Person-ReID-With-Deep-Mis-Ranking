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
from util.utils import fliplr


EPSILON = 8.0 / 255.0
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate AP-Attack against the DukeMTMC-ReID LSRO victim.'
    )
    parser.add_argument('--root', type=str, default='/home/lzf/ldx/datasets')
    parser.add_argument(
        '--victim_weight',
        type=str,
        default=(
            '/home/lzf/ldx/checkpoints/DeepMisRanking/LSRO/duke/'
            'dukemtmcreid.pth.tar'
        ),
    )
    parser.add_argument(
        '--apattack_root', type=str, default='/home/lzf/ldx/projects/AP-Attack'
    )
    parser.add_argument(
        '--generator_weight',
        type=str,
        default=(
            '/home/lzf/ldx/checkpoints/AP-Attack/generator/'
            'apattack_duke_stage2_best_ep60.pth.tar'
        ),
    )
    parser.add_argument('--test_batch', type=int, default=32)
    return parser.parse_args()


def extract_lsro_feature(model, images):
    return model(images, False)[0] + model(fliplr(images), False)[0]


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


def load_generator(apattack_root, generator_weight):
    sys.path.insert(0, apattack_root)
    from advers.GD import Generator

    generator = Generator(3, 3, 32, norm='bn', beta=0.1)
    checkpoint = torch.load(generator_weight, map_location='cpu')
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
    generator.cuda()
    generator.eval()
    return generator


def extract_query_features(loader, model, generator):
    clean_features, adv_features = [], []
    pids, camids = [], []
    max_actual_delta = 0.0
    total_abs_delta = 0.0
    total_delta_values = 0
    attacked_query_count = 0

    for batch_idx, (images, batch_pids, batch_camids, _) in enumerate(loader):
        images = images.cuda()
        clean_features.append(extract_lsro_feature(model, images).cpu())

        adv_images, actual_delta = generate_adversarial_images(generator, images)
        adv_features.append(extract_lsro_feature(model, adv_images).cpu())

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


def extract_gallery_features(loader, model):
    features, pids, camids = [], [], []
    for batch_idx, (images, batch_pids, batch_camids, _) in enumerate(loader):
        images = images.cuda()
        features.append(extract_lsro_feature(model, images).cpu())
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


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this evaluation script.')

    dataset = data_manager.init_img_dataset(
        root=args.root,
        name='dukemtmcreid',
        split_id=0,
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
    )
    opt = get_opts('lsro')
    query_loader = DataLoader(
        ImageDataset(dataset.query, transform=opt['transform_test']),
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=opt['workers'],
        pin_memory=True,
        drop_last=False,
    )
    gallery_loader = DataLoader(
        ImageDataset(dataset.gallery, transform=opt['transform_test']),
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=opt['workers'],
        pin_memory=True,
        drop_last=False,
    )

    model = models.init_model(
        name='lsro',
        pre_dir=args.victim_weight,
        num_classes=dataset.num_train_pids,
        pretrained=False,
    )
    model.cuda()
    model.eval()
    generator = load_generator(args.apattack_root, args.generator_weight)

    with torch.no_grad():
        (
            clean_qf,
            adv_qf,
            q_pids,
            q_camids,
            attacked_query_count,
            max_actual_delta,
            mean_actual_delta,
        ) = extract_query_features(query_loader, model, generator)
        gf, g_pids, g_camids = extract_gallery_features(gallery_loader, model)

    assert attacked_query_count == len(dataset.query)
    _, clean_cmc, clean_mAP = make_results(
        clean_qf,
        gf,
        [],
        [],
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        targetmodel='lsro',
        ak_typ=-1,
    )
    _, adv_cmc, adv_mAP = make_results(
        adv_qf,
        gf,
        [],
        [],
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        targetmodel='lsro',
        ak_typ=-1,
    )

    print_results('CLEAN RESULTS', clean_cmc, clean_mAP)
    print_results('ADVERSARIAL RESULTS', adv_cmc, adv_mAP)
    absolute_mAP_drop = clean_mAP - adv_mAP
    relative_mAP_drop = absolute_mAP_drop / clean_mAP * 100.0
    print('\n=== ATTACK SUMMARY ===')
    print('attacked queries: {}'.format(attacked_query_count))
    print('clean mAP: {:.2%}'.format(clean_mAP))
    print('adv mAP: {:.2%}'.format(adv_mAP))
    print('absolute mAP drop: {:.2%}'.format(absolute_mAP_drop))
    print('relative mAP drop: {:.2f}%'.format(relative_mAP_drop))
    print('\n=== PERTURBATION SANITY ===')
    print('epsilon: {:.8f}'.format(EPSILON))
    print('max |delta_pixel|: {:.8f}'.format(max_actual_delta))
    print('mean |delta_pixel|: {:.8f}'.format(mean_actual_delta))


if __name__ == '__main__':
    main()
