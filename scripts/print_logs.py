import sys
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/print_logs.py <path_to_npy>")
        sys.exit(1)
    p = sys.argv[1]
    arr = np.load(p, allow_pickle=True)
    last = arr[-1]
    # columns: [l_cls, l_dist, l_dist_masked, loss, acc, auc, f1, sp, eo, p0, p1, REF, v0, v1, VEF]
    print(arr.shape)
    print('acc_val=%.4f auc=%.4f f1=%.4f sp=%.4f eo=%.4f REF=%.4f VEF=%.4f' % (
        last[4], last[5], last[6], last[7], last[8], last[11], last[14]
    )) 