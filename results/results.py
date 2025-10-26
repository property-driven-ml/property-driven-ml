from __future__ import print_function

import codecs
import os
import textwrap

import pandas
import numpy

from datetime import timedelta

from pathlib import Path

from collections import namedtuple

Result = namedtuple("Result", "p_acc c_acc c_sec overall")


def format_value(v) -> str:
    return "nan" if v == -1 else rf"\qty{{{v * 100:.2f}}}{{\percent}}"


def get_name_from_file(report: str) -> str:
    name = Path(report).stem

    if name == "Goedel":
        name = "Gödel"
    elif name == "Lukasiewicz":
        name = r"\L ukasiewicz"
    elif name == "KleeneDienes":
        name = "Kleene-Dienes"
    elif name == "ReichenbachSigmoidal":
        name = "sig. Reichenbach"

    if Path(report).stem == "Goedel" and "robustness" in report:
        name = "Fuzzy Logic"

    return name


def get_legendentry_from_file(report: str) -> str:
    return rf"\addlegendentry {{{get_name_from_file(report)}}};"


def write_plot_file(report_dir: str, target_file: str):
    def full_path(r: str) -> str:
        return f"{report_dir}/{r}"

    with codecs.open(target_file, "w", "utf-8") as file:
        begin = (
            textwrap.dedent(r"""
            \documentclass[tikz]{standalone}
            \usepackage[utf8]{inputenc}
            \usepackage[T1]{fontenc}

            \usepackage{amsmath}

            \input{tikz_settings}

            \begin{document}
            \begin{tikzpicture}[font=\small]
              \begin{groupplot}[group/results]
                \nextgroupplot[title={Prediction Accuracy (PAcc)},]
            """).strip()
            + "\n"
        )

        file.write(begin)

        report_files = sorted(
            [
                f
                for f in os.listdir(report_dir)
                if f.endswith(".csv") and not f.endswith("RegressionPlot.csv")
            ]
        )

        def write(report, key: str):
            df = pandas.read_csv(os.path.join(report_dir, report), comment="#")

            # best combination of P-Acc, C-Acc, and C-Sec in the last 10% of epochs
            p_acc = df["Test-P-Metric"].values[-(len(df) // 10) :]
            c_acc = df["Test-C-Acc"].values[-(len(df) // 10) :]
            c_sec = df["Test-C-Sec"].values[-(len(df) // 10) :]
            i = numpy.argmax(p_acc * c_acc * c_sec)

            best_epoch = df["Epoch"].values[-(len(df) // 10) :][i] + 1

            if "Baseline" in report:
                file.write(
                    rf"\addplot+[mark indices={best_epoch}, densely dotted] table [y={key}] {{{full_path(report)}}};"
                    + "\n"
                )
            else:
                file.write(
                    rf"\addplot+[mark indices={best_epoch}] table [y={key}] {{{full_path(report)}}};"
                    + "\n"
                )

        for report in report_files:
            print(f"reading {os.path.join(report_dir, report)}")
            write(report, key="Test-P-Metric")

        intermediate = (
            textwrap.dedent(r"""
            \coordinate (c1) at (rel axis cs:0,1);
                \nextgroupplot[title={Constraint Accuracy (CAcc)},
                  yticklabels={},
                  xlabel={},
                ]
            """).strip()
            + "\n"
        )
        file.write(intermediate)

        for report in report_files:
            write(report, key="Test-C-Acc")

        intermediate = (
            textwrap.dedent(r"""
            \coordinate (c2) at (rel axis cs:0,1);
                \nextgroupplot[title={Constraint Security (CSec)},
                  yticklabel pos=right,
                  yticklabel style={anchor=east,xshift=2.5em},
                  legend to name=full-legend
                ]
            """).strip()
            + "\n"
        )
        file.write(intermediate)

        for report in report_files:
            write(report, key="Test-C-Sec")

        for report in report_files:
            file.write(get_legendentry_from_file(full_path(report)) + "\n")

        end = (
            textwrap.dedent(r"""
            \coordinate (c3) at (rel axis cs:1,1);
              \end{groupplot}
              \coordinate (c4) at ($(c1)!.5!(c3)$);
              \node[below, yshift=0.5cm] at (c4 |- current bounding box.south) {\pgfplotslegendfromname{full-legend}};
            \end{tikzpicture}
            \end{document}
            """).strip()
            + "\n"
        )

        file.write(end)


def write_table_file(report_dir: str, target_file: str):
    def full_path(r: str) -> str:
        return f"{report_dir}/{r}"

    results: dict[str, Result] = {}

    total_seconds = 0.0

    report_files = sorted(
        [
            f
            for f in os.listdir(report_dir)
            if f.endswith(".csv") and not f.endswith("RegressionPlot.csv")
        ]
    )

    if len(report_files) < 1:
        return

    for report in report_files:
        df = pandas.read_csv(os.path.join(report_dir, report), comment="#")

        # best combination of P-Acc, C-Acc, and C-Sec in the last 10% of epochs
        p_acc = df["Test-P-Metric"].values[-(len(df) // 10) :]
        c_acc = df["Test-C-Acc"].values[-(len(df) // 10) :]
        c_sec = df["Test-C-Sec"].values[-(len(df) // 10) :]
        i = numpy.argmax(p_acc * c_acc * c_sec)

        results[full_path(report)] = Result(
            p_acc[i], c_acc[i], c_sec[i], p_acc[i] * c_acc[i] * c_sec[i]
        )

        best = max(results, key=lambda k: results[k].overall)

        avg_test_time = df["Test-Time"].mean()
        avg_train_time = df["Train-Time"].iloc[1:].mean()

        total_test_time = df["Test-Time"].sum()
        total_train_time = df["Train-Time"].iloc[1:].sum()

        total_seconds += total_train_time + total_test_time

        print(
            f"file: {report} avg. train time [s]: {avg_train_time:.2f} avg. test time [s]: {avg_test_time:.2f} total train time [s]: {total_train_time:.2f} total test time [s]: {total_test_time:.2f}"
        )

        with codecs.open(target_file, "w", "utf-8") as file:
            begin = (
                textwrap.dedent(r"""
                \documentclass{standalone}
                \usepackage[utf8]{inputenc}
                \usepackage[T1]{fontenc}

                \usepackage{siunitx}
                \sisetup{detect-all}

                \usepackage{amsmath}
                \usepackage{amssymb}

                \usepackage{tabularray}
                \UseTblrLibrary{booktabs}

                \begin{document}
                \footnotesize
                \begin{tblr}
                  {
                    colspec={Q[l, mode=text]Q[c, mode=text]Q[c, mode=text]Q[c, mode=text]},
                    row{1}={font=\bfseries, mode=text},
                  }
                    \toprule
                      Logic & PAcc & CAcc & CSec \\
                    \midrule
                """).strip()
                + "\n"
            )
            file.write(begin)

            for key, value in results.items():
                if key == best:
                    file.write(
                        rf"{get_name_from_file(key)} & \textbf{{{format_value(value.p_acc)}}} & \textbf{{{format_value(value.c_acc)}}} & \textbf{{{format_value(value.c_sec)}}} \\"
                        + "\n"
                    )
                else:
                    file.write(
                        rf"{get_name_from_file(key)} & {format_value(value.p_acc)} & {format_value(value.c_acc)} & {format_value(value.c_sec)} \\"
                        + "\n"
                    )

            end = (
                textwrap.dedent(r"""
                    \bottomrule
                  \end{tblr}
                \end{document}
                """).strip()
                + "\n"
            )

            file.write(end)

    return total_seconds


def main():
    total_seconds = 0

    for folder_constraint in os.listdir("."):
        if (
            os.path.isdir(folder_constraint)
            and folder_constraint != ".git"
            and folder_constraint != "alsomitra"
        ):
            for folder_dataset in os.listdir(folder_constraint):
                if (
                    os.path.isdir(os.path.join(folder_constraint, folder_dataset))
                    and folder_dataset != ".git"
                ):
                    report_dir = f"{folder_constraint}/{folder_dataset}"

                    plot_file = f"plot_{folder_constraint}_{folder_dataset}.tex"
                    write_plot_file(report_dir, plot_file)

                    table_file = f"table_{folder_constraint}_{folder_dataset}.tex"
                    total_seconds += write_table_file(report_dir, table_file)

    total_time = timedelta(seconds=total_seconds)
    hours, remainder = divmod(total_time.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    print(f"Total time: {total_time.days} days {hours} hours {minutes} minutes")


if __name__ == "__main__":
    main()
