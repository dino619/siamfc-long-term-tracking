import argparse
import os
import cv2

from tools.sequence_utils import VOTSequence
from tools.sequence_utils import save_results
from siamfc import TrackerSiamFC
from siamfc import TrackerSiamFCLongTerm


def load_sequence_names(dataset_path, selected_sequences=None):
    if selected_sequences:
        # Izbira zaporedij iz ukazne vrstice je uporabna pri delu na CPU,
        # ker ni treba spreminjati datoteke list.txt za vsak poskus.
        return selected_sequences
    
    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def create_tracker(args):
    # Obe različici uporabljata isti zagonski skript. Tako so izhodne datoteke
    # enake oblike in jih lahko ocenjevalni skript bere brez posebnih primerov.
    if args.tracker == 'siamfc-lt':
        return TrackerSiamFCLongTerm(
            net_path=args.net,
            redetect_threshold=args.redetect_threshold,
            redetect_samples=args.redetect_samples,
            redetect_batch_size=args.redetect_batch_size,
            redetect_sampling=args.redetect_sampling,
            redetect_sigma=args.redetect_sigma,
            redetect_sigma_growth=args.redetect_sigma_growth,
            redetect_uniform_ratio=args.redetect_uniform_ratio,
            redetect_seed=args.redetect_seed)

    return TrackerSiamFC(net_path=args.net)


def save_sample_regions(sample_regions, file_path):
    # Vzorčene regije shranimo ločeno od glavnih rezultatov. Ocenjevanje bere
    # samo bboxes/scores, vzorci pa so potrebni za razlago ponovne detekcije.
    with open(file_path, 'w') as f:
        for frame_index, boxes in sample_regions:
            boxes_text = ';'.join(
                ','.join([str(el) for el in box]) for box in boxes)
            f.write('%d:%s\n' % (frame_index, boxes_text))


def evaluate_tracker(args):
    
    sequences = load_sequence_names(args.dataset, args.sequence)
    os.makedirs(args.results_dir, exist_ok=True)

    tracker = create_tracker(args)

    for sequence_name in sequences:
        
        print('Processing sequence:', sequence_name)

        bboxes_path = os.path.join(args.results_dir, '%s_bboxes.txt' % sequence_name)
        scores_path = os.path.join(args.results_dir, '%s_scores.txt' % sequence_name)
        states_path = os.path.join(args.results_dir, '%s_states.txt' % sequence_name)
        samples_path = os.path.join(args.results_dir, '%s_samples.txt' % sequence_name)

        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            # Dolgotrajni poskusi so počasni na CPU, zato obstoječe rezultate
            # preskočimo in se izognemo nepotrebnemu ponavljanju.
            print('Results on this sequence already exists. Skipping.')
            continue
        
        sequence = VOTSequence(args.dataset, sequence_name)

        img = cv2.imread(sequence.frame(0))
        gt_rect = sequence.get_annotation(0)
        tracker.init(img, gt_rect)
        results = [gt_rect]
        scores = [[10000]]  # a very large number - very confident at initialization
        # Stanje je dodatna diagnostična informacija: 0 pomeni normalno sledenje,
        # 1 pomeni izgubljeno stanje. Format je ločen, da ne spremeni ocenjevanja.
        states = [[0]]
        sample_regions = []

        if args.visualize:
            cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)
        for i in range(1, sequence.length()):
            if i % args.progress_interval == 0:
                print('Frame %d/%d' % (i, sequence.length()), flush=True)

            img = cv2.imread(sequence.frame(i))
            prediction, score = tracker.update(img)
            results.append(prediction)
            scores.append([score])
            states.append([1 if getattr(tracker, 'lost', False) else 0])
            if args.save_samples and getattr(tracker, 'last_sample_boxes', []):
                # Shranimo samo slike, kjer je bila ponovna detekcija aktivna.
                # To zmanjša datoteko in omogoča jasen prikaz v poročilu.
                sample_regions.append((i, tracker.last_sample_boxes))

            if args.visualize:
                tl_ = (int(round(prediction[0])), int(round(prediction[1])))
                br_ = (int(round(prediction[0] + prediction[2])), int(round(prediction[1] + prediction[3])))
                cv2.rectangle(img, tl_, br_, (0, 0, 255), 1)

                cv2.imshow('win', img)
                key_ = cv2.waitKey(10)
                if key_ == 27:
                    exit(0)
        
        save_results(results, bboxes_path)
        save_results(scores, scores_path)
        save_results(states, states_path)
        if args.save_samples:
            save_sample_regions(sample_regions, samples_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description='SiamFC Runner Script')

    parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
    parser.add_argument("--net", help="Path to the pre-trained network", required=True, action='store')
    parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')
    parser.add_argument("--visualize", help="Show ground-truth annotations", required=False, action='store_true')
    parser.add_argument("--sequence", help="Sequence to evaluate. Can be used multiple times.", required=False, action='append')
    parser.add_argument("--tracker", help="Tracker variant to run", required=False, default='siamfc',
                        choices=['siamfc', 'siamfc-lt'])
    parser.add_argument("--redetect_threshold", help="LT confidence threshold", required=False,
                        type=float, default=4.0)
    parser.add_argument("--redetect_samples", help="Number of sampled regions during re-detection",
                        required=False, type=int, default=96)
    parser.add_argument("--redetect_batch_size", help="Batch size for scoring sampled regions",
                        required=False, type=int, default=32)
    parser.add_argument("--redetect_sampling", help="Sampling strategy used during re-detection",
                        required=False, default='uniform', choices=['uniform', 'gaussian'])
    parser.add_argument("--redetect_sigma", help="Initial Gaussian sampling standard deviation",
                        required=False, type=float, default=80.0)
    parser.add_argument("--redetect_sigma_growth", help="Gaussian sigma multiplier per lost frame",
                        required=False, type=float, default=1.15)
    parser.add_argument("--redetect_uniform_ratio", help="Uniform samples mixed into Gaussian sampling",
                        required=False, type=float, default=0.25)
    parser.add_argument("--redetect_seed", help="Random seed for re-detection sampling",
                        required=False, type=int, default=0)
    parser.add_argument("--save_samples", help="Save sampled re-detection regions for visualization",
                        required=False, action='store_true')
    parser.add_argument("--progress_interval", help="Print progress every N frames",
                        required=False, type=int, default=100)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate_tracker(args)


if __name__ == '__main__':
    main()
