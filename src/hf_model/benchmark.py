import json
import os
import pickle as pkl

import pandas as pd
import plotly.express as px

READ_PATH = (
    "./benchmarks/results/d40998e3504b46c99bece2b8f3dbf174/checkpoints_results.pkl"
)


def read_and_display(read_path: str):
    if not os.path.exists(read_path):
        print(f"Creating {read_path} file")
        os.makedirs(os.path.dirname(read_path), exist_ok=True)
    with open(read_path, "rb") as file:
        results = pkl.load(file)

    # df = pd.DataFrame([{
    # 'ID': item['doc_id'],
    # 'Category': item['doc']['activity_label'],
    # 'Question': item['query'].split(':')[-1].strip(),
    # 'Correct Answer': item['gold'],
    # 'Model Accuracy': item['acc'],
    # 'Top Choice': min(item['filtered_resps'], key=lambda x: x[0])[1],
    # 'Confidence Spread': f"{max(lp for lp, _ in item['filtered_resps']):.1f} to {min(lp for lp, _ in item['filtered_resps']):.1f}"
    # } for item in results])
    print(results["results"])
    # fig = px.bar(
    # pd.DataFrame({
    #     'Model Accuracy': results['acc'],
    #     'Category': results['doc']['activity_label']
    # }),
    # y='Category',
    # x='Model Accuracy',
    # orientation='h',
    # title='Accuracy by Category'
    # )

    # fig.show()


if __name__ == "__main__":
    read_and_display(READ_PATH)
