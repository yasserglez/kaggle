import os
import pprint
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common


def main():
    train_df = common.load_data('train')
    path = [common.OUTPUT_DIR]
    for name in os.listdir(os.path.join(*path)):
        if not os.path.isdir(os.path.join(*path, name)):
            continue
        path.append(name)
        for random_seed in os.listdir(os.path.join(*path)):
            if not os.path.isdir(os.path.join(*path, random_seed)):
                continue
            path.append(random_seed)
            results = []
            for params_str in os.listdir(os.path.join(*path)):
                if not os.path.isdir(os.path.join(*path, params_str)):
                    continue
                path.append(params_str)
                model_results = OrderedDict({'name': name})
                for param in sorted(params_str.split('_')):
                    k, v = param.split('=')
                    k = k.replace('-', '_')
                    model_results[k] = v
                scores = []
                for fold_num in range(1, 11):
                    fold_csv = os.path.join(*path, f'fold{fold_num}_validation.csv')
                    if os.path.isfile(fold_csv):
                        output = pd.read_csv(fold_csv).sort_values('id')
                        target = train_df[train_df['id'].isin(output['id'])].sort_values('id')
                        assert (output['id'].values == target['id'].values).all()
                        output = output[common.LABELS].values
                        target = target[common.LABELS].values
                        score = roc_auc_score(target, output, average='macro')
                        model_results[f'fold{fold_num}'] = score
                        scores.append(score)
                if scores:
                    model_results['mean'] = np.mean(scores)
                    model_results['std'] = np.std(scores)
                results.append(model_results)
                path.pop()
            if results:
                results = pd.DataFrame(results).sort_values('mean', ascending=False)
                results.to_csv(os.path.join(*path, 'evaluation.csv'), index=False)
            path.pop()
        path.pop()


if __name__ == '__main__':
    main()
