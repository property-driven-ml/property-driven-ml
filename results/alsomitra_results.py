from __future__ import print_function

import os
import pandas

from datetime import timedelta

def main():
    total_seconds = 0

    for folder_constraint in sorted(os.listdir('.')):
        if os.path.isdir(folder_constraint) and folder_constraint != '.git':
            for folder_dataset in os.listdir(folder_constraint):
                if os.path.isdir(os.path.join(folder_constraint, folder_dataset)) and folder_dataset != '.git' and folder_dataset == 'alsomitra':
                    report_dir = f'{folder_constraint}/{folder_dataset}'

                    report_files = sorted([f for f in os.listdir(report_dir) if f.endswith('.csv')])

                    if len(report_files) < 1:
                        return
                    
                    best_value = float('inf')
                    best_report = None

                    for report in report_files:
                        df = pandas.read_csv(os.path.join(report_dir, report), comment='#')

                        rmse = df['Test-P-Metric'].values[-1]
                        c_acc = df['Test-C-Acc'].values[-1]
                        c_sec = df['Test-C-Sec'].values[-1]

                        combined = (rmse + (1 - c_acc) + (1 - c_sec))

                        if best_value is None or best_value > combined:
                            best_value = combined
                            best_report = os.path.join(report_dir, report)
                        
                        avg_test_time = df['Test-Time'].mean()
                        avg_train_time = df['Train-Time'].iloc[1:].mean()

                        total_test_time = df['Test-Time'].sum()
                        total_train_time = df['Train-Time'].iloc[1:].sum()

                        total_seconds += (total_train_time + total_test_time)

                        print(f'{os.path.join(report_dir, report)} {rmse} & {c_acc * 100} & {c_sec * 100}')
                        # print(f'{os.path.join(report_dir, report)} RMSE={rmse} C_Acc={c_acc * 100} C_Sec={c_sec * 100} avg. train time [s]: {avg_train_time:.2f} avg. test time [s]: {avg_test_time:.2f} total train time [s]: {total_train_time:.2f} total test time [s]: {total_test_time:.2f}')

                    print(f'best: {best_report}')

    total_time = timedelta(seconds=total_seconds)
    hours, remainder = divmod(total_time.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    print(f'Total time: {total_time.days} days {hours} hours {minutes} minutes')

if __name__ == '__main__':
    main()