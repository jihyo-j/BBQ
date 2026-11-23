# query_bits data: query-side bit-width sweep (efSearch 256, t=1), NQ / Quora Qwen 1,024.
# argv[1] = summary_t1.csv, argv[2] = query_improvement_transitions_t1.csv
# (experiment/nq_quora_qwen1024_query_bitwidth_sweep_t1/). Values copied, not recomputed.
import csv, sys
S = list(csv.DictReader(open(sys.argv[1]))); T = list(csv.DictReader(open(sys.argv[2])))
BITS = [2, 4, 8, 16, 32]
print("# query_bits from", sys.argv[1].split('experiment/')[-1], "and", sys.argv[2].split('experiment/')[-1])
for ds, tag in [("nq_qwen06_1024", "nq"), ("quora_qwen1024", "qu")]:
    for mk, m in [("calibrated_adcbq", "bbq"), ("rabitq_1bit_adaptive_fp", "rabitq")]:
        rows = {int(r['query_bits']): r for r in S if r['dataset'] == ds and r['method'] == mk}
        assert sorted(rows) == BITS
        print(f"\n\n# {m}_{tag}_recall  (x = 1..5 for 2/4/8/16/FP32 bits; query_bits; mean recall@10; qps)")
        for i, b in enumerate(BITS, 1): print(i, b, rows[b]['recall_at_10'], rows[b]['qps'])
        tr = [r for r in T if r['dataset'] == ds and r['method'] == mk]
        tr.sort(key=lambda r: int(r['from_bits'])); assert len(tr) == 4
        print(f"\n\n# {m}_{tag}_improved  (x = 1..4 transition; improved %; worsened %; mean recall change)")
        for i, r in enumerate(tr, 1):
            print(i, 100*float(r['improved_query_fraction']), 100*float(r['worsened_query_fraction']), r['mean_recall_change'])
