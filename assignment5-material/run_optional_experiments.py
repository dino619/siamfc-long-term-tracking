import argparse
import csv
import os
import time
from argparse import Namespace

import numpy as np

from performance_evaluation import evaluate_performance
from run_tracker import evaluate_tracker


def parse_float_list(text):
    return [float(value.strip()) for value in text.split(',') if value.strip()]


def parse_int_list(text):
    return [int(value.strip()) for value in text.split(',') if value.strip()]


def safe_float(value):
    return str(value).replace('.', 'p')


def result_name(sequence, sampling, samples, threshold):
    # Ime mape vsebuje ključne parametre, zato lahko rezultate primerjamo tudi
    # brez odpiranja CSV datoteke.
    return '%s_%s_s%d_t%s' % (
        sequence, sampling, samples, safe_float(threshold))


def result_files(results_dir, sequence):
    return (
        os.path.join(results_dir, '%s_bboxes.txt' % sequence),
        os.path.join(results_dir, '%s_scores.txt' % sequence))


def state_stats(results_dir, sequence):
    # Te statistike merijo vedenje ponovne detekcije, ne samo končni F-score.
    # Uporabne so, ker dva poskusa lahko dobita podoben F-score, a se razlikujeta
    # v tem, koliko časa je sledilnik izgubljen.
    states_path = os.path.join(results_dir, '%s_states.txt' % sequence)
    samples_path = os.path.join(results_dir, '%s_samples.txt' % sequence)

    if not os.path.exists(states_path):
        return {
            'lost_frames': 0,
            'lost_events': 0,
            'recoveries': 0,
            'sample_frames': 0,
            'average_lost_frames_per_event': 0,
            'average_lost_frames_per_recovery': 0,
            'recovery_rate': 0}

    states = np.loadtxt(states_path, delimiter=',').astype(int).reshape(-1)
    # Prehodi 0->1 pomenijo začetek izgubljenega dogodka, prehodi 1->0 pa
    # uspešno vrnitev v normalno sledenje.
    prev = np.concatenate(([0], states[:-1]))
    lost_events = int(np.sum((prev == 0) & (states == 1)))
    recoveries = int(np.sum((prev == 1) & (states == 0)))
    sample_frames = 0
    if os.path.exists(samples_path):
        with open(samples_path, 'r') as f:
            sample_frames = sum(1 for line in f if line.strip())

    lost_frames = int(np.sum(states))
    # Povprečja računamo z zaščito pred deljenjem z nič, da CSV ostane veljaven
    # tudi pri poskusih, kjer sledilnik nikoli ne preide v izgubljeno stanje.
    average_lost_frames_per_event = (
        lost_frames / lost_events if lost_events > 0 else 0)
    average_lost_frames_per_recovery = (
        lost_frames / recoveries if recoveries > 0 else 0)
    recovery_rate = recoveries / lost_events if lost_events > 0 else 0

    return {
        'lost_frames': lost_frames,
        'lost_events': lost_events,
        'recoveries': recoveries,
        'sample_frames': sample_frames,
        'average_lost_frames_per_event': average_lost_frames_per_event,
        'average_lost_frames_per_recovery': average_lost_frames_per_recovery,
        'recovery_rate': recovery_rate}


def make_run_args(args, results_dir, sampling, samples, threshold):
    return Namespace(
        dataset=args.dataset,
        net=args.net,
        results_dir=results_dir,
        visualize=False,
        sequence=[args.sequence],
        tracker='siamfc-lt',
        redetect_threshold=threshold,
        redetect_samples=samples,
        redetect_batch_size=args.batch_size,
        redetect_sampling=sampling,
        redetect_sigma=args.gaussian_sigma,
        redetect_sigma_growth=args.gaussian_sigma_growth,
        redetect_uniform_ratio=args.gaussian_uniform_ratio,
        redetect_seed=args.seed,
        save_samples=args.save_samples,
        progress_interval=args.progress_interval)


def build_configs(args):
    # Vsaka skupina poskusov spremeni samo en tip parametra. To olajša razlago:
    # prag, število vzorcev in tip vzorčenja imajo ločene vrstice v poročilu.
    configs = []

    if args.experiment in ('thresholds', 'all'):
        for threshold in parse_float_list(args.thresholds):
            configs.append(('threshold', args.sampling, args.samples, threshold))

    if args.experiment in ('samples', 'all'):
        for samples in parse_int_list(args.sample_counts):
            configs.append(('samples', args.sampling, samples, args.threshold))

    if args.experiment in ('sampling', 'all'):
        for sampling in ('uniform', 'gaussian'):
            configs.append(('sampling', sampling, args.samples, args.threshold))

    deduped = []
    seen = set()
    for config in configs:
        key = config
        if key in seen:
            continue
        seen.add(key)
        deduped.append(config)
    return deduped


def run_experiments(args):
    os.makedirs(args.results_root, exist_ok=True)
    summary_path = os.path.join(args.results_root, args.summary)
    rows = []

    for experiment, sampling, samples, threshold in build_configs(args):
        name = result_name(args.sequence, sampling, samples, threshold)
        results_dir = os.path.join(args.results_root, name)
        bboxes_path, scores_path = result_files(results_dir, args.sequence)
        run_time = ''

        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            # Če rezultati že obstajajo, jih ponovno uporabimo. To je pomembno,
            # ker CPU zagon sledilnika traja precej dlje kot samo ocenjevanje.
            print('Skipping existing results:', name)
        else:
            print('Running:', name, flush=True)
            start_time = time.time()
            evaluate_tracker(make_run_args(
                args, results_dir, sampling, samples, threshold))
            run_time = time.time() - start_time

        metrics = evaluate_performance(
            args.dataset, results_dir, [args.sequence])
        stats = state_stats(results_dir, args.sequence)
        row = {
            'experiment': experiment,
            'sequence': args.sequence,
            'sampling': sampling,
            'redetect_samples': samples,
            'redetect_threshold': threshold,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f_score': metrics['f_score'],
            'evaluation_threshold': metrics['evaluation_threshold'],
            'lost_frames': stats['lost_frames'],
            'lost_events': stats['lost_events'],
            'recoveries': stats['recoveries'],
            'sample_frames': stats['sample_frames'],
            'average_lost_frames_per_event':
                stats['average_lost_frames_per_event'],
            'average_lost_frames_per_recovery':
                stats['average_lost_frames_per_recovery'],
            'recovery_rate': stats['recovery_rate'],
            'runtime_seconds': run_time,
            'results_dir': results_dir}
        rows.append(row)

    fieldnames = [
        'experiment', 'sequence', 'sampling', 'redetect_samples',
        'redetect_threshold', 'precision', 'recall', 'f_score',
        'evaluation_threshold', 'lost_frames', 'lost_events', 'recoveries',
        'sample_frames', 'average_lost_frames_per_event',
        'average_lost_frames_per_recovery', 'recovery_rate',
        'runtime_seconds', 'results_dir']
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print('Wrote summary:', summary_path)
    print_readable_summary(rows)


def format_metric(value):
    return '%.4f' % float(value)


def print_row(row):
    print(
        '%s | samples=%s | threshold=%s | F=%s | Pr=%s | Re=%s | '
        'lost/event=%s | lost/recovery=%s | recovery_rate=%s' % (
            row['sampling'],
            row['redetect_samples'],
            row['redetect_threshold'],
            format_metric(row['f_score']),
            format_metric(row['precision']),
            format_metric(row['recall']),
            format_metric(row['average_lost_frames_per_event']),
            format_metric(row['average_lost_frames_per_recovery']),
            format_metric(row['recovery_rate'])))


def print_section(title, rows):
    print('')
    print(title)
    if not rows:
        print('  no rows')
        return
    for row in rows:
        print('  ', end='')
        print_row(row)


def print_readable_summary(rows):
    if not rows:
        return

    # Najboljšo nastavitev izberemo po F-score, ker naloga uporablja F-score kot
    # skupno mero med natančnostjo in priklicem.
    best = max(rows, key=lambda row: float(row['f_score']))

    print('')
    print('Readable summary')
    print('Best configuration by F-score:')
    print('  ', end='')
    print_row(best)
    print('  results_dir:', best['results_dir'])

    print_section(
        'Threshold sweep results:',
        [row for row in rows if row['experiment'] == 'threshold'])
    print_section(
        'Sample-count sweep results:',
        [row for row in rows if row['experiment'] == 'samples'])
    print_section(
        'Sampling comparison results:',
        [row for row in rows if row['experiment'] == 'sampling'])


def main():
    parser = argparse.ArgumentParser(
        description='Run optional long-term SiamFC experiments.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--net', required=True)
    parser.add_argument('--results_root', required=True)
    parser.add_argument('--sequence', default='car9')
    parser.add_argument('--experiment', default='all',
                        choices=['thresholds', 'samples', 'sampling', 'all'])
    parser.add_argument('--threshold', type=float, default=4.0)
    parser.add_argument('--thresholds', default='3.5,4.0,4.5')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--sample_counts', default='4,8,16')
    parser.add_argument('--sampling', default='uniform',
                        choices=['uniform', 'gaussian'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gaussian_sigma', type=float, default=80.0)
    parser.add_argument('--gaussian_sigma_growth', type=float, default=1.15)
    parser.add_argument('--gaussian_uniform_ratio', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--summary', default='summary.csv')
    parser.add_argument('--save_samples', action='store_true')
    parser.add_argument('--progress_interval', type=int, default=250)

    run_experiments(parser.parse_args())


if __name__ == '__main__':
    main()
