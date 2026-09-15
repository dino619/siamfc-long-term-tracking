import argparse
import math
import os

import cv2

from show_tracking import read_sample_regions
from tools.sequence_utils import VOTSequence
from tools.sequence_utils import read_results


def read_states(file_path):
    if not os.path.exists(file_path):
        return None
    return [int(row[0]) for row in read_results(file_path)]


def find_redetection_events(states):
    if states is None:
        raise RuntimeError('States file is required to find re-detection events.')

    # Dogodek je neprekinjen interval, kjer je sledilnik izgubljen. Zanimajo nas
    # samo intervali, ki se končajo z vrnitvijo v normalno sledenje.
    events = []
    lost_start = None
    for i, state in enumerate(states):
        if state == 1 and lost_start is None:
            lost_start = i
        elif state == 0 and lost_start is not None:
            events.append({
                'lost_frame': lost_start,
                'recovery_frame': i,
                'duration': i - lost_start})
            lost_start = None

    return events


def select_redetection_event(states, sample_regions):
    events = find_redetection_events(states)
    # Za sliko v poročilu izberemo samo dogodke, kjer imamo shranjene vzorce.
    # Brez teh vzorcev ne moremo pokazati, kako je potekala ponovna detekcija.
    events = [
        event for event in events
        if any(frame in sample_regions
               for frame in range(event['lost_frame'],
                                  event['recovery_frame'] + 1))]

    if not events:
        raise RuntimeError('No recovered lost event with sampled regions was found.')

    # Najdaljši izgubljeni dogodek je praviloma najbolj jasen za poročilo, ker
    # pokaže razliko med neuspehom lokalnega sledenja in kasnejšo ponovno najdbo.
    return max(events, key=lambda event: event['duration'])


def draw_frame(sequence, frame_index, bbox, score, state, sample_regions):
    img = cv2.imread(sequence.frame(frame_index))
    gt = sequence.get_annotation(frame_index)

    # Uporabimo iste barve kot v poročilu: zelena za resnico, rdeča za rezultat,
    # cian za vzorčene regije. Tako slika neposredno podpira razlago v tekstu.
    if not any([math.isnan(el) for el in gt]):
        sequence.draw_region(img, gt, (0, 255, 0), 2)
    for sample_region in sample_regions:
        sequence.draw_region(img, sample_region, (255, 255, 0), 1)
    sequence.draw_region(img, bbox, (0, 0, 255), 2)

    sequence.draw_text(img, 'Frame: %d' % (frame_index + 1), (50, 25))
    sequence.draw_text(img, 'Score: %.3f' % score, (50, 50))
    sequence.draw_text(img, 'State: %s' % state, (50, 75))
    return img


def export_examples(dataset_path, results_dir, sequence_name, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    sequence = VOTSequence(dataset_path, sequence_name)

    # Izvoz bere že shranjene rezultate sledilnika. S tem ne spreminjamo meritev
    # in ne tvegamo drugačnega naključnega vzorčenja pri ponovnem zagonu.
    bboxes = read_results(os.path.join(results_dir, '%s_bboxes.txt' % sequence_name))
    scores = read_results(os.path.join(results_dir, '%s_scores.txt' % sequence_name))
    states = read_states(os.path.join(results_dir, '%s_states.txt' % sequence_name))
    sample_regions = read_sample_regions(
        os.path.join(results_dir, '%s_samples.txt' % sequence_name))

    event = select_redetection_event(states, sample_regions)
    lost_frame = event['lost_frame']
    found_frame = event['recovery_frame']
    exports = [
        (lost_frame, 'lost'),
        (found_frame, 'redetected')]

    output_paths = []

    for frame_index, state_name in exports:
        img = draw_frame(
            sequence,
            frame_index,
            bboxes[frame_index],
            scores[frame_index][0],
            state_name,
            sample_regions.get(frame_index, []))
        output_path = os.path.join(
            output_dir,
            '%s_%s_frame_%06d.jpg' % (prefix, state_name, frame_index + 1))
        cv2.imwrite(output_path, img)
        output_paths.append(output_path)

    print('Selected re-detection event:')
    print('  lost_frame_index:', lost_frame)
    print('  recovered_frame_index:', found_frame)
    print('  duration_frames:', event['duration'])
    print('  lost_score:', scores[lost_frame][0])
    print('  recovered_score:', scores[found_frame][0])
    print('  output_paths:')
    for output_path in output_paths:
        print('   ', output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Export two frames around a long-term re-detection event.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--prefix', default='redetection')

    args = parser.parse_args()
    export_examples(
        args.dataset,
        args.results_dir,
        args.sequence,
        args.output_dir,
        args.prefix)


if __name__ == '__main__':
    main()
