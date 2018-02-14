import os
from collections import OrderedDict

import pandas as pd
from sklearn.metrics import roc_auc_score

import common


def main():
    path = [common.OUTPUT_DIR]
    for name in os.listdir(*path):
        if not os.path.isdir(os.path.join(*path, name)):
            continue
        path.append(name)
        for random_seed in os.listdir(os.path.join(*path)):
            if not os.path.isdir(os.path.join(*path, random_seed)):
                continue
            path.append(random_seed)
            val_df = common.load_data(int(random_seed), 'validation').sort_values('id')
            target = val_df[common.LABELS].values
            results = []
            for params_str in os.listdir(os.path.join(*path)):
                if not os.path.isdir(os.path.join(*path, params_str)):
                    continue
                path.append(params_str)
                if os.path.isfile(os.path.join(*path, 'validation.csv')):
                    model_df = pd.read_csv(os.path.join(*path, 'validation.csv')).sort_values('id')
                    assert (val_df['id'].values == model_df['id'].values).all()
                    model_output = model_df[common.LABELS].values
                    model_results = OrderedDict()
                    model_results['auc'] = roc_auc_score(target, model_output, average='macro')
                    for param in sorted(params_str.split('_')):
                        k, v = param.split('=')
                        k = k.replace('-', '_')
                        model_results[k] = v
                    results.append(model_results)
                path.pop()
            results = pd.DataFrame(results).sort_values('auc', ascending=False)
            results.to_csv(os.path.join(*path, 'evaluation.csv'), index=False)
            path.pop()
        path.pop()


if __name__ == '__main__':
    main()
