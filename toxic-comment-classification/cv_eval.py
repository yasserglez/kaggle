import os
import glob
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd

import common


def main():
    test_file_pattern = '{}/cross_validation/*/test.csv'.format(common.DATA_DIR)
    for test_file in glob.glob(test_file_pattern):
        test_df = pd.read_csv(test_file).sort_values('id')
        target = Variable(torch.FloatTensor(test_df[common.TARGET_COLUMNS].values))
        base_dir = os.path.dirname(test_file)

        for model_name in [f for f in os.listdir(base_dir) if not f.endswith('.csv')]:
            results = []

            for model_file in glob.glob('{}/{}/*.csv'.format(base_dir, model_name)):
                model_df = pd.read_csv(model_file).sort_values('id')
                assert (test_df['id'].values == model_df['id'].values).all()
                model_output = Variable(torch.FloatTensor(model_df[common.TARGET_COLUMNS].values))
                loss = F.binary_cross_entropy(model_output, target).data[0]

                model_results = OrderedDict()
                model_results['loss'] = loss
                for param in sorted(os.path.basename(model_file)[:-4].split('_')):
                    k, v = param.split('=')
                    k = k.replace('-', '_')
                    model_results[k] = v
                results.append(model_results)

            results = pd.DataFrame(results).sort_values('loss')
            results.to_csv('{}/{}.csv'.format(base_dir, model_name), index=False)


if __name__ == '__main__':
    main()
