import argparse
import os
import math
import cv2
import numpy as np

from tools.sequence_utils import VOTSequence
from tools.sequence_utils import read_results


def read_sample_regions(file_path):
    # Vzorci so shranjeni v ločeni datoteki, ker niso del uradne metrike. Za
    # prikaz jih naložimo samo, ko želimo razložiti ponovno detekcijo.
    sample_regions = {}
    if not os.path.exists(file_path):
        return sample_regions

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame_text, boxes_text = line.split(':', 1)
            if not boxes_text:
                sample_regions[int(frame_text)] = []
                continue
            sample_regions[int(frame_text)] = [
                [float(el) for el in box_text.split(',')]
                for box_text in boxes_text.split(';')]

    return sample_regions


def visualize_results(dataset_path, results_dir, sequence_name, show_samples):
    
    sequence = VOTSequence(dataset_path, sequence_name)

    bboxes_path = os.path.join(results_dir, '%s_bboxes.txt' % sequence_name)
    scores_path = os.path.join(results_dir, '%s_scores.txt' % sequence_name)
    states_path = os.path.join(results_dir, '%s_states.txt' % sequence_name)
    samples_path = os.path.join(results_dir, '%s_samples.txt' % sequence_name)

    bboxes = read_results(bboxes_path)
    scores = read_results(scores_path)
    # Stanje in vzorci sta dodatna podatka dolgotrajnega sledilnika. Če ju ni,
    # prikaz še vedno deluje za osnovni SiamFC.
    states = read_results(states_path) if os.path.exists(states_path) else None
    sample_regions = read_sample_regions(samples_path) if show_samples else {}

    if len(sequence.gt) != len(bboxes):
        print('Groundtruth and results does not have the same number of elements.')
        exit(-1)

    overlaps = [sequence.overlap(bb, gt) for bb, gt in zip(bboxes, sequence.gt)]
        
    sequence.initialize_window('Window')

    for i in range(sequence.length()):

        img = cv2.imread(sequence.frame(i))

        gt_ = sequence.get_annotation(i)
        if not any([math.isnan(el) for el in gt_]):
            sequence.draw_region(img, gt_, (0, 255, 0), 2)
        # Vzorčene regije rišemo pred končno napovedjo, da rdeč okvir ostane
        # jasno viden tudi pri več vzorcih.
        for sample_region in sample_regions.get(i, []):
            sequence.draw_region(img, sample_region, (255, 255, 0), 1)
        sequence.draw_region(img, bboxes[i], (0, 0, 255), 2)

        sequence.draw_text(img, '%d/%d' % (i + 1, sequence.length()), (50, 25))
        sequence.draw_text(img, 'Score: %.3f' % scores[i][0], (50, 50))
        sequence.draw_text(img, 'Overlap: %.2f' % overlaps[i], (50, 75))
        if states is not None:
            state = 'Lost' if states[i][0] else 'Tracking'
            sequence.draw_text(img, 'State: %s' % state, (50, 100))

        sequence.show_image(img, 10)


def main():
    parser = argparse.ArgumentParser(description='SiamFC Runner Script')

    parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
    parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')
    parser.add_argument("--sequence", help="Sequence to visualize", required=True, action='store')
    parser.add_argument("--show_samples", help="Visualize sampled re-detection regions", required=False, action='store_true')

    args = parser.parse_args()

    visualize_results(args.dataset, args.results_dir, args.sequence, args.show_samples)


if __name__ == '__main__':
    main()
