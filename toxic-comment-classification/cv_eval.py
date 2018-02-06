import os
import glob
from collections import OrderedDict

import pandas as pd
from sklearn.metrics import roc_auc_score

import common


def main():
    test_file_pattern = '{}/cross_validation/*/test.csv'.format(common.DATA_DIR)
    for test_file in glob.glob(test_file_pattern):
        test_df = pd.read_csv(test_file).sort_values('id')
        target = test_df[common.LABELS].values

        base_dir = os.path.dirname(test_file)
        for model_name in [f for f in os.listdir(base_dir) if not f.endswith('.csv')]:
            results = []
            for model_file in glob.glob('{}/{}/*.csv'.format(base_dir, model_name)):
                model_df = pd.read_csv(model_file).sort_values('id')
                assert (test_df['id'].values == model_df['id'].values).all()
                model_output = model_df[common.LABELS].values

                model_results = OrderedDict()
                model_results['auc'] = roc_auc_score(target, model_output, average='macro')
                for param in sorted(os.path.basename(model_file)[:-4].split('_')):
                    k, v = param.split('=')
                    k = k.replace('-', '_')
                    model_results[k] = v
                results.append(model_results)

            if results:
                results = pd.DataFrame(results).sort_values('auc', ascending=False)
                results.to_csv('{}/{}.csv'.format(base_dir, model_name), index=False)


if __name__ == '__main__':
    main()
